from moviepy.editor import *
import os
import os.path

root_demo_bc = "demonstrations_bc"
root_demo_da = "demonstrations_da"
root_pow_demo_bc = "powerful_demonstrations_bc"
root_pow_demo_da = "powerful_demonstrations_da"

for root in [root_demo_bc, root_demo_da]:
    dirs = os.listdir(path=root)
    for directory in dirs:
        if not os.path.isdir(root+"/"+directory):
            continue
        files = os.listdir(path=root+"/"+directory)
        for f in files:
            if f[-4:] == '.mp4':
                clip = (VideoFileClip(root+"/"+directory+"/"+f)
                        .subclip(0,8)
                        .resize(0.4))
                clip.write_gif(root+"/"+directory+"/"+directory+".gif", fps=15)
