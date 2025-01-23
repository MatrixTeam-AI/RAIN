import numpy as np
import torch
from src.morpher import RAINMorpher
import cv2
from src.modeling.translation import translation
a = RAINMorpher("./configs/rain_morpher.yaml", torch.device("cuda:0"),)

import gradio as gr
import numpy as np
import cv2
from collections import deque
import time
import os

webrtc_mode = False # Set to True requires gradio-webrtc==0.0.20

res = np.zeros((512, 512, 3) , dtype=np.uint8)
timestamp = -1.0
cropping = 0.9
work_flag = False
return_bulk = 16
counter_max = 100
temporary_video_clips_folder = "./video_clips_tmp"

params = dict(face_length_ratio=0.75,
eye_height_ratio=3.0,
eye_width_ratio=1.6,
mouth_height_ratio=1.2,
mouth_width_ratio=0.67,
jaw_width_ratio=0.9,
eye_horizontal_correction=0.15,
eye_distance_correction=0.05,
eye_vertical_correction=0.10,
mouth_horizontal_correction=0.15,
mouth_vertical_correction=0.03,
nose_horizontal_correction=0.1,
nose_vertical_correction=0.05,
)

def transform_func(frame):
    global timestamp
    global res
    if not(a.work_flag):
        return res
    h, w = frame.shape[:2]
    d = int(min(h, w) * cropping)
    mh = (h - d) // 2
    mw = (w - d) // 2

    frame = frame[mh:h-mh,mw:w-mw,:]
    if(webrtc_mode):
        frame = frame[:,:,::-1]
    a.push_input(frame) 
    current_timestamp = time.time()
    if(current_timestamp - timestamp >= 1.0 / (a.fps + len(a.output_pile) * 2 )):
        u = a.fetch_one_frame()
        if not (u is None):
            res = u
            if (webrtc_mode):
                res = (255.0 * res).astype(np.uint8)
                res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            timestamp = current_timestamp
    return res

def fuse_ref(img):
    if(a.vae is None):
        return "Please load reference part first"
    if(img is None):
        return "Please upload reference image"
    a.fuse_reference(img)
    return "Finished"

def unload_ref():
    a.load_reference_part_to_cpu_or_uninstall()
    return "Unloaded successfully"

def reload_ref():
    a.load_reference_part_to_device()
    return "Reloaded successfully"

def clear_ref():
    if(a.tensorrt):
        return "TensorRT Mode does not support this operation"
    a.clear_reference()
    return "Succesfully cleared reference"

def clear_queue():
    a.clear_queue()
    return "Successfully cleared queue"

def change_fps(fps_):
    a.set_fps(fps_)

def change_cropping(inp):
    global cropping
    cropping = inp

def start_process():
    global work_flag
    if(work_flag):
        return
    a.kick_start(0.01)
    work_flag = True

def stop_process():
    global work_flag
    a.stop()
    a.clear_queue()
    work_flag = False

def render_pose(frame):
    if(frame is None):
        return None
    h, w = frame.shape[:2]
    d = int(min(h, w) * cropping)
    mh = (h - d) // 2
    mw = (w - d) // 2
    frame = frame[mh:h-mh,mw:w-mw,:]
    return a.face_tracker.detect(frame)

def set_translations(frame):
    a.set_translations(param2translations(**params))
    if(frame is None):
        return None
    h, w = frame.shape[:2]
    d = int(min(h, w) * cropping)
    mh = (h - d) // 2
    mw = (w - d) // 2
    frame = frame[mh:h-mh,mw:w-mw,:]
    return a.face_tracker.detect(frame)
    
def set_params(name):
    def func(value, frame):
        global params
        global a
        params[name] = value
        a.set_translations(param2translations(**params))
        if(frame is None):
            return None
        h, w = frame.shape[:2]
        d = int(min(h, w) * cropping)
        mh = (h - d) // 2
        mw = (w - d) // 2
        frame = frame[mh:h-mh,mw:w-mw,:]
        return a.face_tracker.detect(frame)

    return func


def start_offline_process(vid):
    global work_flag
    global cropping
    if(work_flag):
        return
    
    vc = cv2.VideoCapture(vid)
    rval = vc.isOpened()    
    if(rval):
        rval, frame = vc.read()

    h, w = frame.shape[:2]
    d = int(min(h, w) * cropping)
    mh = (h - d) // 2
    mw = (w - d) // 2

    a.push_input(cv2.resize(frame[mh:h-mh, mw:w-mw, ::-1], (512, 512)), eager_mode=True)

    while rval:
        rval, frame = vc.read()
        if not(frame is None):
            a.push_input(cv2.resize(frame[mh:h-mh, mw:w-mw, ::-1], (512, 512)), eager_mode=True)

    a.kick_start(0.001)
    work_flag = True
    video_clips = []
    counter = 0
    if not os.path.exists(temporary_video_clips_folder):
        os.mkdir(temporary_video_clips_folder)
    while work_flag:
        result = a.fetch_one_frame()
        if not (result is None):
            video_clips.append(result)
        if len(video_clips) >= return_bulk:
            video_codec = cv2.VideoWriter_fourcc(*"mp4v")
            name = f"{temporary_video_clips_folder}/output-{counter}.mp4"
            name_ts = f"{temporary_video_clips_folder}/output-{counter}.ts"
            if(os.path.exists(name)):
                os.remove(name)
            if(os.path.exists(name_ts)):
                os.remove(name_ts)
            segment_file = cv2.VideoWriter(name, video_codec, a.fps, video_clips[0].shape[:2])
            for frame in video_clips:
                segment_file.write(cv2.cvtColor(np.uint8(255.0 * frame), cv2.COLOR_BGR2RGB))
            segment_file.release()
            yield name
            counter += 1
            if(counter == counter_max):
                counter = 0
            video_clips = []
        time.sleep(0.01)

def stop_offline_process():
    global work_flag
    a.stop()
    a.clear_queue()
    work_flag = False

def param2translations(face_length_ratio=0.75, eye_height_ratio=3.0, eye_width_ratio=1.6, mouth_height_ratio=1.2, mouth_width_ratio=0.67, jaw_width_ratio=0.9, eye_horizontal_correction=0.15, eye_distance_correction=0.05, eye_vertical_correction=0.10, mouth_horizontal_correction=0.15, mouth_vertical_correction=0.0, nose_vertical_correction=0.0,nose_horizontal_correction=0.0):
    translations = ([
        translation(0, 68, vertical_scale=face_length_ratio, xaxis=(2, 14)),
        translation(36, 42,vertical_scale=eye_height_ratio, xaxis=(0,3), horizontal_scale=eye_width_ratio,),
        translation(42, 48,vertical_scale=eye_height_ratio, xaxis=(0,3), horizontal_scale=eye_width_ratio),
        translation(60, 68,vertical_scale=mouth_height_ratio, xaxis=(0, 4), horizontal_scale=mouth_width_ratio),
        translation(36, 42, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=eye_horizontal_correction - eye_distance_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(33, 13), horizontal_transition=eye_horizontal_correction),
        translation(36, 42, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-eye_horizontal_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(33, 3), horizontal_transition=-(eye_horizontal_correction - eye_distance_correction)),
        translation(36, 42, xaxis=(0,3), relative_length=(19, 6), vertical_transition=eye_vertical_correction),
        translation(42, 48, xaxis=(0,3), relative_length=(24, 10), vertical_transition=eye_vertical_correction),
        translation(60, 68, xaxis=(0,4), relative_length=(33, 13), horizontal_transition=mouth_horizontal_correction),
        translation(60, 68, xaxis=(0,4), relative_length=(33, 3), horizontal_transition=-mouth_horizontal_correction),
        translation(60, 68, relative_length=(27, 8), xaxis=(0, 4), vertical_transition=mouth_vertical_correction),
        translation(27, 36, relative_length=(27, 8), xaxis=(4, 8), vertical_transition=nose_vertical_correction),
        translation(27, 36, relative_length=(22, 16), xaxis=(4, 8), horizontal_transition=nose_horizontal_correction),
        translation(27, 36, relative_length=(21, 0), xaxis=(4, 8), horizontal_transition=-nose_horizontal_correction),
        translation(4, 13, xaxis=(0,8), horizontal_scale=jaw_width_ratio),
    ])
    return translations

css_="""
"""
with gr.Blocks(css=css_) as demo:
    gr.HTML("<h2>Reference Image</h2><br/><p>Upload the reference image first (or choose from examples), you can safely unload reference module after fusing for saving device memory. We recommend images with clear background, half-portraits and anime characters.</p>")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(elem_classes=["reference_image"])
            btn = gr.Button("Fuse Reference")
        with gr.Column():
            btn1 = gr.Button("Unload Reference Module")
            btn2 = gr.Button("Reload Reference Module")
            btn3 = gr.Button("Clear Reference")
            btn4 = gr.Button("Clear Queue")
            inp1 = gr.Slider(minimum=0.1,maximum=30.0,label="FPS",value=a.fps)
            inp1.change(fn=change_fps, inputs=inp1)
            inp2 = gr.Slider(minimum=0.0,maximum=1.0,label="cropping",value=cropping, elem_id="cropping_setter")
            inp2.change(fn=change_cropping, inputs=inp2)
            out = gr.Textbox(show_label=False, placeholder="STATE",interactive=False)
        btn.click(fn=fuse_ref, inputs=inp, outputs=out)
        btn1.click(fn=unload_ref, outputs=out)
        btn2.click(fn=reload_ref, outputs=out)
        btn3.click(fn=clear_ref, outputs=out)
        btn4.click(fn=clear_queue, outputs=out)
    gr.Examples(
        [["./example_images/test1.png"], ["./example_images/test2.png"], ["./example_images/test3.png"], ["./example_images/test4.png"], ["./example_images/test5.png"], ["./example_images/test6.png"], ["./example_images/test7.png"], ["./example_images/test8.png"], ["./example_images/test9.png"], ["./example_images/test10.png"]],
        [inp],
        cache_examples=False,
    )
    gr.HTML("<h2>Face Morphing</h2><br/><p>Start webcam and click on 'start' for morphing, click on stop for stoping. The input will be centerly cropped according to cropping ratio. You can change the reference image while morphing if device memory is enough, or you will have to stop and reload reference module, fuse reference, unload reference module and then restart. There will be delay in reaction.</p>")
    with gr.Row():
        if webrtc_mode:
            from gradio_webrtc import WebRTC
            with gr.Column(elem_classes=["my-column"]):
                track_constraints = {"width": {"max": 1920}, "height": {"max": 1080}, "resizeMode": "none"}
                rtp_params = {"degradationPreference": "maintain-resolution", "priority": "high"}
                image = WebRTC(label="Stream", mode="send-receive", modality="video", track_constraints=track_constraints, rtp_params=rtp_params)
            with gr.Column():
                btn5 = gr.Button("Start")
                btn6 = gr.Button("Stop")
            image.stream(
                fn=transform_func,
                inputs=[image], # 
                outputs=[image], time_limit=1000, concurrency_limit = 30
            )
            btn5.click(fn=start_process)
            btn6.click(fn=stop_process)
        else:
            with gr.Column():
                input_img = gr.Image(sources=["webcam"], type="numpy", streaming=True)
                btn5 = gr.Button("Start")
            with gr.Column():
                output_img = gr.Image( type="numpy", streaming=True)
                btn6 = gr.Button("Stop")
            input_img.stream(transform_func, [input_img], [output_img], time_limit=30, stream_every=1.0 / (2 * a.fps), concurrency_limit=30)
            btn5.click(fn=start_process)
            btn6.click(fn=stop_process)

    gr.HTML("<h2>Face Calibration</h2><br/><p>Take on a photo of yourself and then calibrate the params for facical landmarks morphing. You can try this until you feel good enough. </p>")
    with gr.Row():
        with gr.Column():
            input_img_pose = gr.Image()
            im_output_pose = gr.Image(interactive=False)
            btn8 = gr.Button("Render")
            btn8.click(fn=render_pose, inputs=input_img_pose, outputs=im_output_pose)
        with gr.Column():
            inp3 = gr.Slider(minimum=0.1, maximum=1.5, label="Face Length Ratio",value=params["face_length_ratio"], step=0.01)
            inp4 = gr.Slider(minimum=0.1, maximum=4.0, label="Eye Height Ratio",value=params["eye_height_ratio"], step=0.01)
            inp5 = gr.Slider(minimum=0.1, maximum=3.0, label="Eye Width Ratio",value=params["eye_width_ratio"], step=0.01)
            inp6 = gr.Slider(minimum=0.1, maximum=3.0, label="Mouth Height Ratio",value=params["mouth_height_ratio"], step=0.01)
            inp7 = gr.Slider(minimum=0.1, maximum=3.0, label="Mouth Width Ratio",value=params["mouth_width_ratio"], step=0.01)
            inp8 = gr.Slider(minimum=0.1, maximum=1.5, label="Jaw Width Ratio",value=params["jaw_width_ratio"], step=0.01)
            inp10 = gr.Slider(minimum=-0.25, maximum=0.25, label="Eye Distance Fix",value=params["eye_distance_correction"], step=0.01)
        with gr.Column():
            inp11 = gr.Slider(minimum=-0.25, maximum=0.25, label="Eye Vertical Fix",value=params["eye_vertical_correction"], step=0.01)
            inp9 = gr.Slider(minimum=-0.25, maximum=0.25, label="Eye Horizontal Fix (Fix eye offset when facing sideways)",value=params["eye_horizontal_correction"], step=0.01)
            inp12 = gr.Slider(minimum=-0.25, maximum=0.25, label="Mouth Horizontal Fix (Fix mouth offset when facing sideways)",value=params["mouth_horizontal_correction"], step=0.01)
            inp13 = gr.Slider(minimum=-0.25, maximum=0.25, label="Mouth Vertical Fix",value=params["mouth_vertical_correction"], step=0.01)
            inp14 = gr.Slider(minimum=-0.25, maximum=0.25, label="Nose Horizontal Fix (Fix nose offset when facing sideways)",value=params["nose_horizontal_correction"], step=0.01)
            inp15 = gr.Slider(minimum=-0.25, maximum=0.25, label="Nose Vertical Fix",value=params["nose_vertical_correction"], step=0.01)
        inp3.change(set_params("face_length_ratio"),[inp3, input_img_pose], [im_output_pose])
        inp4.change(set_params("eye_height_ratio"),[inp4, input_img_pose], [im_output_pose])
        inp5.change(set_params("eye_width_ratio"),[inp5, input_img_pose], [im_output_pose])
        inp6.change(set_params("mouth_height_ratio"),[inp6, input_img_pose], [im_output_pose])
        inp7.change(set_params("mouth_width_ratio"),[inp7, input_img_pose], [im_output_pose])
        inp8.change(set_params("jaw_width_ratio"),[inp8, input_img_pose], [im_output_pose])
        inp9.change(set_params("eye_horizontal_correction"),[inp9, input_img_pose], [im_output_pose])
        inp10.change(set_params("eye_distance_correction"),[inp10, input_img_pose], [im_output_pose])
        inp11.change(set_params("eye_vertical_correction"),[inp11, input_img_pose], [im_output_pose])
        inp12.change(set_params("mouth_horizontal_correction"),[inp12, input_img_pose], [im_output_pose])
        inp13.change(set_params("mouth_vertical_correction"),[inp13, input_img_pose], [im_output_pose])
        inp14.change(set_params("nose_horizontal_correction"),[inp14, input_img_pose], [im_output_pose])
        inp15.change(set_params("nose_vertical_correction"),[inp15, input_img_pose], [im_output_pose])
        
    gr.HTML("<h2>Offline Rendering</h2><br/><p>Upload a video and get generated video in an offline rendering manner (Recommend for non-TensorRT mode).</p>")
    with gr.Row():
        with gr.Column():
            input_vid_offline = gr.Video()
            btn9 = gr.Button("Start")

        with gr.Column():
            offline_output = gr.Video(label="Output",interactive=False, format="mp4", loop=False, streaming=True)
            btn10 = gr.Button("Stop")
        btn9.click(fn=start_offline_process, inputs=input_vid_offline, outputs=offline_output)
        btn10.click(fn=stop_offline_process)
        
    

demo.launch()