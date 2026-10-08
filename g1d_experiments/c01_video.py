"""Exact indexed RGB decoding through PyAV, independent of OpenCV's FFmpeg build."""
from fractions import Fraction
import av


def frames_at(path, indices):
    wanted = list(map(int, indices))
    if wanted != sorted(set(wanted)) or (wanted and wanted[0] < 0):
        raise ValueError('Frame indices must be unique, nonnegative and sorted')
    if not wanted:
        return
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        cursor = 0
        for index, frame in enumerate(container.decode(stream)):
            if index == wanted[cursor]:
                yield frame.to_ndarray(format='rgb24')
                cursor += 1
                if cursor == len(wanted):
                    return
    raise ValueError(f'Missing recorded frame: {path}/{wanted[cursor]}')


def frame_at(path, index):
    # Seek to the preceding keyframe and identify the target by presentation time.
    # If a video lacks usable timing metadata, sequential decoding remains exact.
    index = int(index)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate
        origin = stream.start_time or 0
        if rate and stream.time_base:
            target = origin + int(Fraction(index, 1) / rate / stream.time_base)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    break
                actual = round((frame.pts-origin)*stream.time_base*rate)
                if actual == index:
                    return frame.to_ndarray(format='rgb24')
                if actual > index:
                    break
    return next(frames_at(path, [index]))
