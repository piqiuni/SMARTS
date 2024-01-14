import os
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter


def record_video():
    fig = plt.figure()
    metadata = dict(title='title', artist='Matplotlib',comment='')
    writer = FFMpegWriter(fps=10, metadata=metadata)
    
    
    video_name = 'video_name' + '.mp4'

    with writer.saving(fig, video_name, 300):
        for i in range(0, 10):
            xs = list(range(0, i+1))
            ys = list(range(0, 2*i+1, 2))
            plt.plot(xs,ys)
            fig.canvas.draw()
            # plt.pause(0.1)
            time.sleep(0.1)
            print(i)
            writer.grab_frame()
            # print()
        # plt.close()
    print("record ok")
    
record_video()