# rescue_stt_node.py v0.211 2026-03-17
# [이번 버전에서 수정된 사항]
# - UI direct-subscribe 토픽(/robot6/victim_voice_reply, /robot6/tts/done)과 /robot6/tts/request 구독의 durability를 VOLATILE로 통일
# - /robot/stop 수신 시 진행 중인 STT 시나리오를 즉시 중단하는 기존 동작 유지
# - 기존 siren_path 탐색, 임시 TTS 파일 처리 방식은 유지
# - /robot6/tts/request 계약이 상태 문자열임을 주석으로 명확화
# - 런타임 필수 외부 의존성(오디오 장치/파이썬 패키지) 안내 추가

import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import rclpy
import speech_recognition as sr
import torch
import whisper
from gtts import gTTS
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from std_msgs.msg import Bool, String

'''
================================================================================
[Rescue Dialogue Node]
ROS 2 환경에서 구조 대상자와 상호작용하기 위한 음성 인식/합성 노드.
Mic 입력으로 피해자의 상황을 파악하고, TTS와 사이렌으로 응답 및 구조를 진행합니다.

토픽 계약:
- 구독: /robot6/tts/request (std_msgs/String)
  - 페이로드는 JSON이 아니라 상태 문자열(NORMAL, CAUTION, WARNING, CRITICAL)
- 발행: /robot6/tts/done (std_msgs/Bool)
  - 구조 대화 또는 NORMAL 스킵 처리 종료 후 True 발행

런타임 필수 조건:
- Python 패키지: pygame, speech_recognition, torch, whisper, gtts, numpy
- 장치/환경: 마이크 입력, 오디오 출력, siren mp3 파일(선택이지만 권장)
================================================================================
'''


class RescueDialogueNode(Node):
    def __init__(self):
        super().__init__('rescue_dialogue_node')

        ui_event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        '기존 음성 기록 퍼블리셔'
        self.publisher_ = self.create_publisher(String, '/robot6/victim_voice_reply', ui_event_qos)

        '행동 제어용 keep_going 토픽'
        self.keep_going_pub = self.create_publisher(String, '/robot6/keep_going', 10)

        'Control 노드 연동용 토픽'
        self.tts_done_pub = self.create_publisher(Bool, '/robot6/tts/done', ui_event_qos)
        self.tts_req_sub = self.create_subscription(
            String,
            '/robot6/tts/request',
            self.tts_request_callback,
            ui_event_qos,
        )
        self.stop_sub = self.create_subscription(Bool, '/robot/stop', self.stop_callback, 10)

        '실행 상태 잠금용 플래그'
        self.is_running = False
        self._stop_requested = False

        '사이렌 경로 파라미터 선언'
        self.declare_parameter('siren_path', '')
        self.siren_path = self._resolve_siren_path()
        if self.siren_path:
            self.get_logger().info(f'사이렌 파일 경로 확인 완료: {self.siren_path}')
        else:
            self.get_logger().warn('사이렌 파일을 찾지 못했습니다. siren_path 파라미터 또는 파일 위치를 확인하세요.')

        self.get_logger().info('AI 시스템(Whisper) 로딩 중... 잠시만 기다려주세요.')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = whisper.load_model('small').to(self.device)
        self.r = sr.Recognizer()
        self.mic = sr.Microphone()

        with self.mic as source:
            self.r.adjust_for_ambient_noise(source, duration=1)

        self.get_logger().info('시스템 준비 완료. /robot6/tts/request 토픽 대기 중...')

    def _resolve_siren_path(self):
        '사이렌 파일 경로를 여러 후보 위치에서 탐색'
        param_path = self.get_parameter('siren_path').get_parameter_value().string_value.strip()
        env_path = os.environ.get('RESCUE_SIREN_PATH', '').strip()

        script_path = Path(__file__).resolve()
        script_dir = script_path.parent
        candidates = []

        if param_path:
            candidates.append(Path(param_path).expanduser())

        if env_path:
            candidates.append(Path(env_path).expanduser())

        candidates.append(Path.cwd() / 'siren.mp3')
        candidates.append(script_dir / 'siren.mp3')

        '패키지 내부나 워크스페이스 상위 디렉토리까지 탐색'
        for parent in [script_dir, *script_dir.parents[:6]]:
            candidates.append(parent / 'siren.mp3')

        candidates.append(Path.home() / 'rokey_ws' / 'siren.mp3')
        candidates.append(Path('/home/jaylee/Downloads/siren.mp3'))

        checked = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            resolved_str = str(resolved)
            if resolved_str in checked:
                continue
            checked.add(resolved_str)
            if resolved.exists() and resolved.is_file():
                return resolved_str

        return ''

    def _play_audio_file(self, audio_path, label):
        '지정된 오디오 파일을 재생하고 실패 시 상세 로그를 남김'
        if not audio_path or not os.path.exists(audio_path):
            self.get_logger().error(f'[{label}] 오디오 파일을 찾을 수 없습니다: {audio_path}')
            return False

        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init()

            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_requested and label == 'TTS':
                    pygame.mixer.music.stop()
                    return False
                time.sleep(0.1)
            return True
        except Exception as e:
            self.get_logger().error(f'[{label}] 오디오 재생 실패: {e}')
            return False
        finally:
            try:
                if pygame.mixer.get_init() is not None:
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
            except Exception as cleanup_error:
                self.get_logger().warn(f'[{label}] 오디오 정리 중 경고: {cleanup_error}')

    def stop_callback(self, msg: Bool):
        if not msg.data:
            return
        self._stop_requested = True
        self.is_running = False
        self.get_logger().warn('[Stop] /robot/stop 수신 -> 진행 중인 시나리오 중단.')

    def speak(self, text):
        print(f"\n[구급 로봇 🔊]: '{text}'")

        temp_path = ''
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name

            tts = gTTS(text=text, lang='ko')
            tts.save(temp_path)
            success = self._play_audio_file(temp_path, 'TTS')
            if not success:
                self.get_logger().warn('TTS 음성 출력에는 실패했지만 시나리오는 계속 진행합니다.')
        except Exception as e:
            self.get_logger().error(f'TTS 생성 또는 재생 실패: {e}')
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    self.get_logger().warn(f'임시 TTS 파일 삭제 실패: {e}')

    def play_siren(self):
        if not self.siren_path:
            self.siren_path = self._resolve_siren_path()

        if not self.siren_path:
            self.get_logger().error('사이렌 파일 경로를 확인하지 못해 재생할 수 없습니다.')
            return False

        print('\n🚨 [비상] 응답 없음! 사이렌을 울립니다. 🚨')
        return self._play_audio_file(self.siren_path, 'SIREN')

    def listen(self):
        print('🎤 듣는 중...')
        try:
            with self.mic as source:
                audio = self.r.listen(source, timeout=7)
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            wav_data = np.frombuffer(raw_data, np.int16).flatten().astype(np.float32) / 32768.0
            result = self.model.transcribe(wav_data, language='ko')
            text = result['text'].strip()
            if not text or re.search('[a-zA-Z]', text):
                return ''
            return text
        except sr.WaitTimeoutError:
            self.get_logger().warn('제한 시간 내 음성 입력이 없어 응답 없음으로 처리합니다.')
            return None
        except Exception as e:
            self.get_logger().error(f'STT 처리 실패: {e}')
            return None

    def tts_request_callback(self, msg: String):
        'Control 노드로부터 상태 신호가 오면 실행되는 콜백'
        status = msg.data.strip().upper()
        self.get_logger().info(f"[request 수신] 상태: '{status}'")

        if status == 'ANALYZING':
            self.get_logger().info('  -> 상태 분석 중... (무시)')
            return

        if status == 'NORMAL':
            self.get_logger().info('  -> NORMAL 상태: 대화 생략, done 토픽 발행')
            done_msg = Bool()
            done_msg.data = True
            self.tts_done_pub.publish(done_msg)
            return

        if self.is_running:
            self.get_logger().warn('이미 대화 시나리오가 실행 중입니다. 추가 요청을 무시합니다.')
            return

        if status in ['CAUTION', 'WARNING', 'CRITICAL']:
            self.get_logger().info(f'  -> {status} 상태: 구조 대화 시나리오 시작!')
            self.is_running = True
            self._stop_requested = False
            thread = threading.Thread(target=self.run_scenario, daemon=True)
            thread.start()
            return

        self.get_logger().warn(f"  -> 알 수 없는 상태 메시지입니다: '{status}'")

    def run_scenario(self):
        self._stop_requested = False
        try:
            self.speak('괜찮으세요?')
            if self._stop_requested:
                return
            reply_1 = self.listen()
            if self._stop_requested:
                return

            if reply_1:
                print(f'👤 1차 응답: {reply_1}')
                self.speak('어디가 안 좋으신가요?')
                if self._stop_requested:
                    return
                reply_2 = self.listen()
                if self._stop_requested:
                    return
                final_msg = f"1차 응답: {reply_1} / 2차 응답: {reply_2 if reply_2 else '없음'}"

                '2차 응답이 정상적으로 들어왔을 때 keep_going 토픽 발행'
                if reply_2:
                    keep_msg = String()
                    keep_msg.data = 'keep_going'
                    self.keep_going_pub.publish(keep_msg)
                    self.get_logger().info('  -> 2차 응답 완료')
                else:
                    final_msg = '긴급!!!!!!!!!!!!!!!!!!!'
                    self.play_siren()
            else:
                final_msg = '긴급!!!!!!!!!!!!!!!!!'
                self.play_siren()

            if self._stop_requested:
                return
            self.speak('물과 구급상자가 있고 곧 구조대가 올 것입니다.')
            if self._stop_requested:
                return

            msg = String()
            msg.data = final_msg
            self.publisher_.publish(msg)

            '관제 노드에 임무 종료 보고'
            done_msg = Bool()
            done_msg.data = True
            self.tts_done_pub.publish(done_msg)
            print('✅ STT 시나리오 종료 및 관제 노드에 보고 완료.')
        except Exception as e:
            self.get_logger().error(f'시나리오 실행 중 예외 발생: {e}')
        finally:
            self.is_running = False


def main(args=None):
    rclpy.init(args=args)
    node = RescueDialogueNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
