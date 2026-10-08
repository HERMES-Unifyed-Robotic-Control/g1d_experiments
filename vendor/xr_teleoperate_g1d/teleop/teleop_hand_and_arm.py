import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO, 
                       file=True, 
                       file_path="/home/unitree/unitree_eai_environment/logs",
                       backup_count=100,
                       max_file_size=50*1024*1024,
                       file_name_format="{prog_name}_%Y%m%d.log",
                       )
logger_mp = logging_mp.getLogger(__name__)
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import queue
import json
import numpy as np
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from teleop.televuer.tv_wrapper import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_Arm_Internal_Dex1_Controller
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller, Dex1_1_Gripper_Controller
from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX, Inspire_Controller_FTP, Inspire_Controller_DFX_ctrl, Inspire_Controller_FTP_ctrl
from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand, Brainco_Controller_ctrl
from teleop.robot_control.mobile_control import G1_Mobile_Lift_Controller
from teleop.utils.instruction_map import ControlDataMapper, HandleInstruction
from xr_teleoperate_g1d.unitree_teleop_adapter import from_televuer

from teleop.teleimager.src.teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
# from teleop.utils.motion_switcher import MotionSwitcher
from sshkeyboard import listen_keyboard, stop_listening
try:
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except ImportError:
    AudioClient = None

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
EPISODE_ID     = 0      # Episode ID (int) for IPC communication


class VoiceAnnouncer:
    """Non-blocking Mandarin/English announcements through the G1 voice service."""

    def __init__(self, enabled=True, language="zh"):
        self.enabled = enabled and AudioClient is not None
        self.language = language
        self.messages = queue.Queue(maxsize=16)
        self.running = self.enabled
        self.client = None
        if self.enabled:
            try:
                self.client = AudioClient()
                self.client.SetTimeout(2.0)
                self.client.Init()
                # sdk2py 1.x increments this value incorrectly when it starts
                # at zero; seed it so consecutive TTS requests stay unique.
                self.client.tts_index = 1
                self.worker = threading.Thread(target=self._run, daemon=True)
                self.worker.start()
            except Exception as e:
                self.enabled = False
                self.running = False
                logger_mp.warning(f"Voice announcer unavailable: {e}")

    def say(self, zh_text, en_text):
        if not self.enabled:
            return
        messages = []
        if self.language in ("zh", "bilingual"):
            messages.append((zh_text, 0))
        if self.language in ("en", "bilingual"):
            messages.append((en_text, 1))
        for text, speaker_id in messages:
            try:
                self.messages.put_nowait((text, speaker_id))
            except queue.Full:
                logger_mp.warning(f"Voice queue full; dropped announcement: {text}")

    def _run(self):
        while self.running or not self.messages.empty():
            try:
                text, speaker_id = self.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ret = self.client.TtsMaker(text, speaker_id)
                if ret != 0:
                    logger_mp.warning(f"TTS failed with code {ret}: {text}")
                else:
                    # TtsMaker starts playback asynchronously.  Leave enough
                    # time before the next bilingual sentence so it is not
                    # cut off by the following request.
                    if speaker_id == 0:
                        duration = len(text) / 7.0
                    else:
                        duration = len(text.split()) / 2.5
                    time.sleep(max(1.0, min(5.0, duration)))
            except Exception as e:
                logger_mp.warning(f"TTS error: {e}")

    def close(self):
        self.running = False
        if self.enabled and hasattr(self, "worker"):
            self.worker.join(timeout=2.5)


class ControllerShortcutMapper:
    """Map Pico face-button rising edges to the R/S/Q state-machine actions."""

    def __init__(self, announcer):
        self.announcer = announcer
        self.previous = {"right_a": False, "right_b": False, "left_y": False}

    def update(self, tele_data):
        if not tele_data.motion_data_ready:
            return
        current = {
            "right_a": bool(tele_data.right_ctrl_aButton),
            "right_b": bool(tele_data.right_ctrl_bButton),
            # Vuer exposes the left controller's X/Y pair as aButton/bButton.
            "left_y": bool(tele_data.left_ctrl_bButton),
        }
        if current["right_a"] and not self.previous["right_a"]:
            on_press("r")
            self.announcer.say(
                "右手柄诶键。遥操作已启动",
                "Right controller A button. Teleoperation started.",
            )
            logger_mp.info("Pico right A -> R: teleoperation started")
        if current["left_y"] and not self.previous["left_y"]:
            # READY is false for the whole time an episode is open.  A stop
            # request must therefore be accepted while RECORD_RUNNING is true;
            # READY only gates creation of the next episode.
            if START and not RECORD_TOGGLE and (RECORD_RUNNING or READY):
                starting = not RECORD_RUNNING
                on_press("s")
                if starting:
                    self.announcer.say(
                        "左手柄歪键。开始采集",
                        "Left controller Y button. Recording started.",
                    )
                else:
                    self.announcer.say(
                        "左手柄歪键。结束采集，正在保存",
                        "Left controller Y button. Recording stopped. Saving data.",
                    )
                logger_mp.info("Pico left Y -> S: recording toggled")
            elif START:
                self.announcer.say(
                    "数据正在处理，请稍候",
                    "Data is being processed. Please wait.",
                )
            else:
                self.announcer.say(
                    "请先按右手柄诶键启动遥操作",
                    "Press the right controller A button to start teleoperation first.",
                )
        if current["right_b"] and not self.previous["right_b"]:
            on_press("q")
            self.announcer.say(
                "右手柄比键。遥操作结束",
                "Right controller B button. Teleoperation stopped.",
            )
            logger_mp.info("Pico right B -> Q: exit requested")
        self.previous = current
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: READY starts an episode; RECORD_RUNNING always permits stopping it.
#  --> auto  : Auto-transition after saving data.

def on_press(key, episode_id=None):
    global STOP, START, RECORD_TOGGLE, EPISODE_ID
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        if episode_id is not None:
            EPISODE_ID = episode_id
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1', 'H2', 'R1'], default='G1', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex1_internal', 'dex3', 'brainco', 'inspire_ftp', 'inspire_dfx'], help='Select end effector controller')
    # mobile base, elevation and waist control
    parser.add_argument('--base-type', type=str, choices=['mobile_lift', 'lift','legs'], default='mobile_lift', help='Select lower body type')
    parser.add_argument('--use-waist', action = 'store_true', help = 'Enable waist control')
    # mode flags
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--no-voice', action='store_true', help='Disable status announcements')
    parser.add_argument('--voice-language', choices=['zh', 'en', 'bilingual'], default='zh', help='Status announcement language')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server')
    parser.add_argument('--xr-webrtc-host', type=str, default=None, help='Image-server address reachable by the XR headset; defaults to img-server-ip')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = '/home/unitree/unitree_eai_environment/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    announcer = None
    try:
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        announcer = VoiceAnnouncer(enabled=not args.no_voice and not args.sim, language=args.voice_language)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.xr_webrtc_host or args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw" # another choice is "head_position".
                                     )

        # Enter debug mode
        # motion_switcher = MotionSwitcher()
        # status, result = motion_switcher.Enter_Debug_Mode()
        # logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")
        
        xr_motion_data_ready = Value('b', False, lock=True)
        # arm
        if args.arm == "G1":
            arm_ik = G1_29_ArmIK()
            if args.ee == "dex1_internal":
                left_gripper_value = Value('d', 0.0, lock=True)        # [input]
                right_gripper_value = Value('d', 0.0, lock=True)       # [input]
                dual_gripper_data_lock = Lock()
                dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
                dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
                arm_ctrl = G1_29_Arm_Internal_Dex1_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                                                              dual_gripper_state_array, dual_gripper_action_array,
                                                              simulation_mode=args.sim, use_waist=args.use_waist,
                                                              xr_motion_data_ready_in=xr_motion_data_ready)
            else:
                arm_ctrl = G1_29_ArmController(simulation_mode=args.sim, use_waist=args.use_waist)
        elif args.arm == "H2":
            logger_mp.warning("H2 arm is not supported yet.")
            raise NotImplementedError("H2 arm is not supported yet.")
        elif args.arm == "R1":
            logger_mp.warning("R1 arm is not supported yet.")
            raise NotImplementedError("R1 arm is not supported yet.")

        # end-effector
        if args.ee == "dex3":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1": # external
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                    dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass

        # For mobile base and elevation control
        if args.base_type != "legs":
            try:
                mobile_ctrl = G1_Mobile_Lift_Controller(args.base_type, args.input_mode == "hand")
            except Exception as e:
                STOP = True
                logger_mp.error(f"Failed to initialize mobile base/lift controller: {e}")
                raise
        else:
            mobile_ctrl=None
        control_data_mapper = ControlDataMapper(arm_ctrl.get_current_waist_q()[0])
        handle_instruction = HandleInstruction(args.input_mode == "hand", tv_wrapper, mobile_ctrl)
        controller_shortcuts = ControllerShortcutMapper(announcer)

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)
            save_pending = False
            pending_episode_id = None
            pending_episode_path = None

        logger_mp.info("Please enter the start signal (enter 'r' to start the subsequent program)")
        announcer.say(
            "系统准备完成，请进入虚拟现实。右手柄诶键启动遥操作，左手柄歪键开始采集，右手柄比键退出",
            "System ready. Enter virtual reality. Press right A to start, left Y to record, and right B to exit.",
        )
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            if args.input_mode == "controller":
                controller_shortcuts.update(tv_wrapper.get_tele_data())
            time.sleep(0.033)

        logger_mp.info("---------------------🚀start program🚀-------------------------")
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record:
                    head_img = img_client.get_head_frame()
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode(episode_id=EPISODE_ID if args.ipc else None):
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    pending_episode_id = recorder.episode_id
                    pending_episode_path = recorder.json_path
                    queued_frames = recorder.item_data_queue.qsize()
                    recorder.save_episode()
                    save_pending = True
                    logger_mp.info(
                        f"==> Episode {pending_episode_id:04d} recording stopped; "
                        f"flushing {queued_frames} queued frame(s) to {pending_episode_path}"
                    )
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
            
            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if args.input_mode == "controller":
                controller_shortcuts.update(tele_data)
                if STOP:
                    break
            normalized_xr = from_televuer(tele_data)
            handle_instruction_data = handle_instruction.get_instruction(tele_data)
            if not normalized_xr.motion_ready:
                control_data_mapper.stop_motion()

            # logger_mp.info(f"tele_data: {tele_data}")
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready
            
            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            # current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()
            left_arm_pose_state, right_arm_pose_state = arm_ik.solve_fk(current_lr_arm_q)
            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q)
            left_arm_pose_action, right_arm_pose_action = arm_ik.solve_fk(sol_q)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

            # For mobile base and elevation control
            height_state = None
            height_action = [0.0]
            move_state = None
            move_action = [0.0, 0.0]
            waist_state = None
            waist_action = None
            if  mobile_ctrl is not None:
                height_state = mobile_ctrl.g1_height_state_array_out
                vel_data = control_data_mapper.update(ry=handle_instruction_data['ry'])
                height_action = np.array([vel_data['g1_height']]).tolist()
                mobile_ctrl.g1_height_action_array_in[0] = height_action[0]  
                if args.base_type == "mobile_lift":
                    move_state = mobile_ctrl.g1_move_state_array_out
                    vel_data = control_data_mapper.update(lx=handle_instruction_data['lx'], ly=handle_instruction_data['ly'])
                    move_action = np.array([vel_data['mobile_x_vel'], vel_data['mobile_yaw_vel']]).tolist()
                    mobile_ctrl.g1_move_action_array_in[0] = move_action[0]  
                    mobile_ctrl.g1_move_action_array_in[1] = move_action[1] 

            if args.use_waist:
                waist_state = arm_ctrl.get_current_waist_q()

                vel_data = control_data_mapper.update(rx=handle_instruction_data['rx'], current_waist_pitch=waist_state[0])
                waist_action = np.array([vel_data['waist_pitch_pos']], dtype=float)
                
                sol_q = np.concatenate([sol_q, waist_action])

            try:   
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            except Exception as e:
                logger_mp.error(f"Failed to control arms with ik solution: {e}")
                raise e
            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                if save_pending and READY:
                    try:
                        with open(pending_episode_path, "r", encoding="utf-8") as episode_file:
                            saved_episode = json.load(episode_file)
                        saved_frames = len(saved_episode.get("data", []))
                        saved_bytes = os.path.getsize(pending_episode_path)
                        logger_mp.info(
                            f"==> Episode {pending_episode_id:04d} save verified: "
                            f"{saved_frames} frame(s), {saved_bytes / (1024 * 1024):.2f} MiB, "
                            f"path={pending_episode_path}"
                        )
                        announcer.say(
                            f"第{pending_episode_id}段采集保存完成，共{saved_frames}帧，可以开始下一段",
                            f"Episode {pending_episode_id} saved with {saved_frames} frames. Ready for the next episode.",
                        )
                    except Exception as e:
                        logger_mp.error(
                            f"Episode {pending_episode_id:04d} save verification failed: {e}"
                        )
                        announcer.say(
                            f"第{pending_episode_id}段保存校验失败，请查看终端",
                            f"Episode {pending_episode_id} save verification failed. Check the terminal.",
                        )
                    save_pending = False
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee in ("dex1", "dex1_internal") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee in ("brainco", "inspire_dfx", "inspire_ftp") and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[7:7+7]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },       
                        "left_arm_pose": {
                            "qpos": left_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_state.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                  
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 

                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },     
                        "left_arm_pose": {
                            "qpos": left_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },
                        "right_arm_pose": {
                            "qpos": right_arm_pose_action.tolist(),
                            "qvel": [],
                            "torque": [],
                        },                     
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 

                        
                    }
                    # Keep raw operator input and explicit G1-D commands in the
                    # same EpisodeWriter frame as robot feedback and images.
                    states["xr"] = normalized_xr.to_dict()
                    actions["operator"] = {
                        "chassis": {
                            "vx": float(move_action[0]),
                            "vy": 0.0,
                            "vyaw": float(move_action[1]),
                        },
                        "lift": {"vz": float(height_action[0])},
                        "waist": {
                            "pitch_q": float(waist_action[0]) if waist_action is not None else None,
                        },
                        "dex1": {
                            "left_closed_ratio": normalized_xr.left_trigger,
                            "right_closed_ratio": normalized_xr.right_trigger,
                        },
                        "arm": {
                            "motion_ready": normalized_xr.motion_ready,
                            "left_grip": normalized_xr.left_grip,
                            "right_grip": normalized_xr.right_grip,
                            "left_wrist_pose": normalized_xr.left_pose,
                            "right_wrist_pose": normalized_xr.right_pose,
                        },
                    }
                    if mobile_ctrl is not None:
                        states["torso"] = {
                            "height": np.array(height_state[0]).tolist(),
                            "qvel": np.array(height_state[1]).tolist()
                        }
                        actions["torso"] = {
                            "qvel": np.array(height_action[0]).tolist()
                        }
                        if args.base_type == "mobile_lift":
                            states["chassis"] = {
                                "qvel": np.array(move_state).tolist()  # [x_vel, yaw_vel]
                            }
                            actions["chassis"] = {
                                "qvel": np.array(move_action).tolist()   # [x_vel, yaw_vel]
                            }
                    if args.use_waist and waist_state is not None and waist_action is not None:
                        states["waist"] = {
                            "qpos": waist_state.tolist(),  # [pitch]
                        }
                        actions["waist"] = {
                            "qpos": waist_action.tolist(),  # [pitch]
                        }

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception as e:
        logger_mp.error(f"Error: {e}")
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        # try:
        #     if not args.motion:
        #         status, result = motion_switcher.Exit_Debug_Mode()
        #         logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        # except Exception as e:
        #     logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        try:
            if announcer is not None:
                announcer.close()
        except Exception as e:
            logger_mp.error(f"Failed to close voice announcer: {e}")
        logger_mp.info("Finally, exiting program.")
