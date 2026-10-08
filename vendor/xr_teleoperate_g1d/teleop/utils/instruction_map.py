import numpy as np

class HandleInstruction:
    def __init__(self, r3_controller, tv_wrapper, mobile_ctrl):
        self.r3_controller = r3_controller
        self.tv_wrapper = tv_wrapper
        self.mobile_ctrl = mobile_ctrl
    def get_instruction(self, tele_data=None):
        if self.r3_controller and self.mobile_ctrl is not None:
            lx = self.mobile_ctrl.r3_controller_state_array_out[0]
            ly = -self.mobile_ctrl.r3_controller_state_array_out[1]
            rx = -self.mobile_ctrl.r3_controller_state_array_out[2]
            ry = -self.mobile_ctrl.r3_controller_state_array_out[3]
            rbutton_A = True if int(self.mobile_ctrl.r3_controller_state_array_out[4]) == 256 else False
            rbutton_B = True if int(self.mobile_ctrl.r3_controller_state_array_out[4]) == 512 else False
        else:
            # Reuse the main-loop sample so arm, base, lift, waist and recorded
            # XR data all describe the same instant.
            tele_data = tele_data if tele_data is not None else self.tv_wrapper.get_tele_data()
            if tele_data.motion_data_ready:
                lx = -tele_data.left_ctrl_thumbstickValue[1]
                ly = -tele_data.left_ctrl_thumbstickValue[0]
                rx = -tele_data.right_ctrl_thumbstickValue[0]
                ry = -tele_data.right_ctrl_thumbstickValue[1]
                rbutton_A = tele_data.right_ctrl_aButton
                rbutton_B = tele_data.right_ctrl_bButton
            else:
                lx = ly = rx = ry = 0.0
                rbutton_A = rbutton_B = False
        return {'lx': lx, 'ly': ly, 'rx': rx, 'ry': ry, 'rbutton_A': rbutton_A, 'rbutton_B': rbutton_B}

class LowPassFilter:
    """Low-pass filter for smoothing data"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value

    def reset(self):
        self._value = 0.0
        self._last_value = 0.0


class ControlDataMapper:
    """
    Control data mapper for mobile base and elevation
    """
    def __init__(self, current_waist_pitch=None):
        # Velocity filters
        self._filters = {
            'mobile_x_vel': LowPassFilter(alpha=0.15),
            'mobile_yaw_vel': LowPassFilter(alpha=0.15)
        }
        
        # Height accumulated value (remains unchanged after release)
        self.height_speed_value = 0
        self.mobile_x_vel = 0
        self.mobile_yaw_vel = 0
        self.current_waist_pitch_pos = current_waist_pitch if current_waist_pitch is not None else 0.0

    def stop_motion(self):
        """Immediately clear velocity commands after XR tracking is lost."""
        for filter_ in self._filters.values():
            filter_.reset()
        self.mobile_x_vel = 0.0
        self.mobile_yaw_vel = 0.0
        self.height_speed_value = 0.0

    def update(self, lx=None, ly=None, rx=None, ry=None, current_waist_pitch=None):
        if lx is not None:
            # Map forward velocity 
            raw = self._map_forward_velocity(lx)
            mobile_x_vel = self._filters['mobile_x_vel'].update(raw, max_accel=1.0)
            self.mobile_x_vel = mobile_x_vel
        else:
            mobile_x_vel = self.mobile_x_vel

        if ly is not None:
            # G1-D is differential drive: left-stick X maps to yaw velocity.
            raw = self._map_yaw_velocity(ly)
            mobile_yaw_vel = self._filters['mobile_yaw_vel'].update(raw, max_accel=1.0)
            self.mobile_yaw_vel = mobile_yaw_vel
        else:
            mobile_yaw_vel = self.mobile_yaw_vel

        # Right-stick X incrementally controls G1-D waist pitch. Limits are the
        # product limits, expressed in radians: -2.5 deg to +135 deg.
        if rx is not None and current_waist_pitch is not None:
            waist_pitch_pos = self._update_waist_position(
                rx,
                current_waist_pitch,
                max_step=0.01,
                min_position=np.deg2rad(-2.5),
                max_position=np.deg2rad(135.0),
            )
        elif current_waist_pitch is not None:
            waist_pitch_pos = self.current_waist_pitch_pos
        else:
            waist_pitch_pos = self.current_waist_pitch_pos

        self.height_speed_value = self._smooth_map(ry, -1.0, 1.0) if ry is not None else 0.0
        
        return {
            'mobile_x_vel': mobile_x_vel,
            'mobile_yaw_vel': mobile_yaw_vel,
            'waist_pitch_pos': waist_pitch_pos,
            'g1_height': self.height_speed_value,
        }

    def _update_waist_position(self, raw_value, current_position, max_step, min_position, max_position):
        step = self._smooth_map(raw_value, -max_step, max_step, deadzone=0.1)
        if step != 0.0:
            self.current_waist_pitch_pos = float(current_position) + step
            self.current_waist_pitch_pos = np.clip(self.current_waist_pitch_pos, min_position, max_position)
        else:
            self.current_waist_pitch_pos = float(current_position)

        return self.current_waist_pitch_pos
    
    def _map_forward_velocity(self, value):
        return self._smooth_map(value, -0.2, 0.2)
    
    def _map_yaw_velocity(self, value):
        return self._smooth_map(value, -0.6, 0.6)
    
    def _smooth_map(self, value, out_min, out_max, deadzone=0.05):
        """
        Smooth mapping function
        Maps input value to output range using deadzone and smooth curve
        
        Args:
            value: Input value (-1 to 1)
            out_min: Output minimum value
            out_max: Output maximum value
            deadzone: Deadzone size
        """
        if abs(value) < deadzone:
            return 0.0
        t = (abs(value) - deadzone) / (1.0 - deadzone)
        t = np.clip(t, 0.0, 1.0)
        smooth = 6 * t**5 - 15 * t**4 + 10 * t**3
        return smooth * (out_max if value > 0 else out_min)
