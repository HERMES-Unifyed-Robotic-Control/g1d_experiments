"""G1D gravity and read-only DDS telemetry. Importing this module sends no commands."""
from pathlib import Path
import argparse
import hashlib
import json
import threading
import time
import numpy as np

ARM_JOINT_NAMES = tuple(
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')
)
DEFAULT_URDF = '/home/unitree/xr_teleoperate_g1d/assets/g1_D/g1_d.urdf'


class GravityCompensation:
    """Same fixed-waist, 14-arm-DOF reduction and static RNEA as collection IK."""
    def __init__(self, urdf=DEFAULT_URDF, scale=1.0):
        import pinocchio as pin
        self.pin = pin
        self.urdf = Path(urdf)
        self.scale = float(scale)
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError('gravity scale must be within [0,1]')
        full = pin.buildModelFromUrdf(str(self.urdf))
        missing = [n for n in ARM_JOINT_NAMES if not full.existJointName(n)]
        if missing:
            raise ValueError(f'G1D URDF missing arm joints: {missing}')
        locked = [j for j in range(1,full.njoints) if full.names[j] not in ARM_JOINT_NAMES]
        self.model = pin.buildReducedModel(full, locked, pin.neutral(full))
        names = tuple(self.model.names)[1:]
        if names != ARM_JOINT_NAMES or self.model.nq != 14 or self.model.nv != 14:
            raise ValueError(f'Unexpected gravity model joint order: {names}')
        self.data = self.model.createData()
        self.last_tau = np.zeros(14)

    def compute(self, arm_q):
        q = np.asarray(arm_q,dtype=np.float64)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError('gravity input must be finite 14D arm qpos')
        tau = np.asarray(self.pin.rnea(self.model,self.data,q,np.zeros(14),np.zeros(14))).copy()*self.scale
        if not np.isfinite(tau).all() or np.any(abs(tau)>self.model.effortLimit):
            raise ValueError('invalid gravity torque or URDF effort limit exceeded')
        self.last_tau = tau
        return tau

    def metadata(self):
        return {'urdf':str(self.urdf),'urdf_sha256':hashlib.sha256(self.urdf.read_bytes()).hexdigest(),
                'joint_order':list(ARM_JOINT_NAMES),'scale':self.scale,'method':'RNEA(q,0,0), other joints locked at zero'}


class RobotTelemetry:
    """Only subscribes to state topics; never creates a command publisher."""
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import Point32_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        self.height_sub = ChannelSubscriber('rt/hispeed_state',Point32_)
        self.low_sub = ChannelSubscriber('rt/lowstate',LowState_)
        self.height_sub.Init();self.low_sub.Init()
        self._lock = threading.Lock();self._stop = threading.Event()
        self._height = None;self._low = None
        self._thread = threading.Thread(target=self._poll,daemon=True,name='g1d-readonly-telemetry')
        self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            height = self.height_sub.Read();low = self.low_sub.Read();now=time.monotonic()
            with self._lock:
                if height is not None:self._height=(float(height.y),float(height.z),now)
                if low is not None:
                    self._low=([float(low.motor_state[i].q) for i in range(15,29)],
                               float(low.motor_state[12].q),float(low.motor_state[14].q),now)
            self._stop.wait(.005)

    def snapshot(self,require_fresh=False):
        now=time.monotonic()
        with self._lock:
            height=self._height;low=self._low
        result={'height_m':height[0] if height else None,'height_velocity_m_s':height[1] if height else None,
                'height_age_s':now-height[2] if height else None,
                'arm_q':low[0] if low else None,'waist_yaw_rad':low[1] if low else None,
                'waist_pitch_rad':low[2] if low else None,'lowstate_age_s':now-low[3] if low else None}
        if require_fresh and (height is None or low is None or now-height[2]>.5 or now-low[3]>.5):
            raise RuntimeError(f'Missing/stale read-only height or lowstate: {result}')
        return result

    def wait_ready(self,timeout=5.0):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try:return self.snapshot(require_fresh=True)
            except RuntimeError:time.sleep(.02)
        return self.snapshot(require_fresh=True)

    def wait_height(self,target,tolerance=.003,timeout=15.0,stable_seconds=.5):
        end=time.monotonic()+timeout;stable=None
        while time.monotonic()<end:
            state=self.snapshot(require_fresh=True);now=time.monotonic()
            within=abs(state['height_m']-target)<=tolerance and abs(state['height_velocity_m_s'])<=.005
            stable=now if within and stable is None else (stable if within else None)
            if stable is not None and now-stable>=stable_seconds:return state
            time.sleep(.02)
        raise RuntimeError(f'Height did not settle at {target:.3f}m within {timeout}s: {self.snapshot()}')

    def close(self):
        self._stop.set();self._thread.join(timeout=1)
        for sub in (self.height_sub,self.low_sub):
            close=getattr(sub,'Close',None)
            if close is not None:close()


class GravityAlignedController:
    """Apply gravity to every ready/interpolated arm target via the existing controller."""
    def __init__(self,controller,gravity):
        self._controller=controller;self.gravity=gravity

    def __getattr__(self,name):
        return getattr(self._controller,name)

    @property
    def arm_velocity_limit(self):return self._controller.arm_velocity_limit

    @arm_velocity_limit.setter
    def arm_velocity_limit(self,value):self._controller.arm_velocity_limit=value

    @property
    def running(self):return self._controller.running

    @running.setter
    def running(self,value):self._controller.running=value

    def ctrl_dual_arm(self,q_target,tauff_target):
        q=np.asarray(q_target,dtype=np.float64)
        return self._controller.ctrl_dual_arm(q,self.gravity.compute(q[:14]))


def main():
    parser=argparse.ArgumentParser(description='Read actual G1D height/waist/arm and compute gravity; no commands.')
    parser.add_argument('--network-interface',default='eth0')
    parser.add_argument('--urdf',default=DEFAULT_URDF)
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(0,networkInterface=args.network_interface)
    telemetry=RobotTelemetry()
    try:
        state=telemetry.wait_ready();gravity=GravityCompensation(args.urdf)
        report={'measured_state':state,'gravity_tau_nm':gravity.compute(state['arm_q']).tolist(),
                'gravity_model':gravity.metadata(),'read_only':True}
        print(json.dumps(report,indent=2),flush=True)
        if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    finally:telemetry.close()


if __name__=='__main__':main()
