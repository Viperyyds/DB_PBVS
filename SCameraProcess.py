from utils.pose_estimation import get_rigid_transform_o3d, get_T_from_rt_vec, trans_pts, get_rt_vec_from_T, \
    get_T_from_rt_rpy, get_rt_rpy_from_T, vec_to_rpy, get_trans_error, get_trvec_rpy_from_T
from utils.utils import get_time_acc
import cv2
import time
from multiprocessing import Process, Pipe, Queue
from pyapriltags import Detector
from pypylon import pylon
from utils.tag_detection import apriltag_two_stage, get_tag_results_dict, get_tag_board_pose, get_tag_corners_3d_pts, \
    apriltag_two_stage_en
from utils.stereo_triangulation import  get_reproj_error_stereo
import numpy as np
from queue import Empty, Full

camera_features_file = r'./camera_config/acA4112-8gmNPE_40212367_cam0.pfs'
class SCameraProcess(Process):
    def __init__(self, camera_calib_params, flower_tag_board_params, db_tag_id_lst, cam_exp_time, pipe, result_queue):
        super(SCameraProcess, self).__init__()
        self.camera_calib_params = camera_calib_params
        self.flower_tag_board_params = flower_tag_board_params
        self.db_tag_id_lst = db_tag_id_lst
        self.out_pipe, self.in_pipe = pipe
        self.result_queue = result_queue
        self.update_num = 10
        self.exp_time = cam_exp_time

    def run(self):
        at_detector = Detector(families='tag25h9',
                               nthreads=16,
                               quad_decimate=1.5,
                               quad_sigma=0.0,
                               refine_edges=True,
                               decode_sharpening=0.0,
                               debug=False
                               )
        at_detector_re = Detector(families='tag25h9',
                                  nthreads=8,
                                  quad_decimate=1.0,
                                  quad_sigma=0.0,
                                  refine_edges=True,
                                  decode_sharpening=0.0,
                                  debug=False
                                  )
        # 连接相机
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")
        cameras = pylon.InstantCameraArray(min(len(devices), 2))
        l = cameras.GetSize()
        if l < 2:
            print("error camera num less than 2!")
            exit(-1)
        cam0 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
        cam0.Open()
        pylon.FeaturePersistence.Load(camera_features_file, cam0.GetNodeMap())
        cam0.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        cam1 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[1]))
        cam1.Open()
        pylon.FeaturePersistence.Load(camera_features_file, cam1.GetNodeMap())
        cam1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # 导入相机标定参数
        stereo_calb_params = self.camera_calib_params
        K0 = stereo_calb_params['M1']
        dist0 = stereo_calb_params['dist1']

        K1 = stereo_calb_params['M2']
        dist1 = stereo_calb_params['dist2']

        proc_ratio = 1.2
        tag_board_size = 55
        det_tags = []
        db_tag_ids = self.db_tag_id_lst
        duban_tag_width = 56.1
        duban_tag_height = 56.1

        save_stereo_images_flag = False

        tag_board_obj_pts = get_tag_corners_3d_pts(tag_board_size, tag_board_size)
        duban_tag_obj_pts = get_tag_corners_3d_pts(duban_tag_width, duban_tag_height)

        cv2.namedWindow('show_img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('show_img', 800, 600)

        while cam0.IsGrabbing() and cam1.IsGrabbing():
            st0 = time.time()
            grabResult0 = cam0.RetrieveResult(7000, pylon.TimeoutHandling_ThrowException)
            grabResult1 = cam1.RetrieveResult(7000, pylon.TimeoutHandling_ThrowException)
            tag_poses_3d_pts = {}
            if grabResult0.GrabSucceeded() and grabResult1.GrabSucceeded():
                # 获取图像数据
                img0 = grabResult0.GetArray()
                img1 = grabResult1.GetArray()

                show_img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)

                cam0_tags = apriltag_two_stage_en(at_detector, img0, proc_ratio, exter_size=21, tag_size=tag_board_size,
                                                  det_tags=det_tags, at_detector_re=at_detector_re)
                cam0_tags_dict = get_tag_results_dict(cam0_tags)

                cam1_tags = apriltag_two_stage_en(at_detector, img1, proc_ratio, exter_size=21, tag_size=tag_board_size,
                                                  det_tags=det_tags, at_detector_re=at_detector_re)
                cam1_tags_dict = get_tag_results_dict(cam1_tags)

                for tag_id in cam0_tags_dict:
                    if tag_id in cam1_tags_dict:
                        tag_corners0 = cam0_tags_dict[tag_id].corners
                        tag_corners1 = cam1_tags_dict[tag_id].corners
                        tag_corners0_e = np.vstack(
                            [tag_corners0, cam0_tags_dict[tag_id].center.reshape(1, 2)])
                        tag_corners1_e = np.vstack(
                            [tag_corners1, cam1_tags_dict[tag_id].center.reshape(1, 2)])
                        mean_error_proj, tag_3d_pts = get_reproj_error_stereo(tag_corners0_e, tag_corners1_e,
                                                                              stereo_calb_params)
                        tag_3d_ref_pts = tag_board_obj_pts
                        if tag_id in db_tag_ids:
                            tag_3d_ref_pts = duban_tag_obj_pts
                        tag_stereo_poses = get_rigid_transform_o3d(tag_3d_ref_pts, tag_3d_pts)
                        trans_e = get_trans_error(tag_3d_ref_pts, tag_3d_pts, tag_stereo_poses)
                        if trans_e < 2:
                            tag_poses_3d_pts[tag_id] = [tag_3d_pts.copy(), tag_stereo_poses.copy(),
                                                        tag_corners0_e.copy()]
                flower_cam_pose = get_tag_board_pose(tag_poses_3d_pts, self.flower_tag_board_params)
                output_data = [flower_cam_pose, tag_poses_3d_pts, get_time_acc()]
                if save_stereo_images_flag:
                    output_data = [flower_cam_pose, tag_poses_3d_pts, get_time_acc(), img0, img1]
                if flower_cam_pose is not None:
                    tcp_cam_pose = flower_cam_pose
                    tcp_rvec, tcp_tvec = get_rt_vec_from_T(tcp_cam_pose)
                    cv2.drawFrameAxes(show_img0, K0, dist0, tcp_rvec, tcp_tvec / 1000, 0.08, 10)
                show_img = show_img0.copy()
                cv2.imshow('show_img', show_img)
                wait_key = cv2.waitKey(1)
                if wait_key == ord('q') or wait_key == 27:
                    output_data = [flower_cam_pose, tag_poses_3d_pts, -1]
                    if save_stereo_images_flag:
                        output_data = [flower_cam_pose, tag_poses_3d_pts, -1, img0, img1]
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except Empty as e:
                            pass
                    try:
                        self.result_queue.put_nowait(output_data)
                    except Full as e:
                        pass
                    break
                # 清理图像序列
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Empty as e:
                        pass
                try:
                    self.result_queue.put_nowait(output_data)
                except Full as e:
                    pass
            grabResult0.Release()
            grabResult1.Release()
            msg = ''
            if self.out_pipe.poll():
                msg = self.out_pipe.recv()
            if msg == f"stereo camera close":
                print(f'stereo camera closing...')
                msg = ''
                break
            elif msg == f"update target pose":
                hole_poses_cam = []
                det_tags = []
                msg = ''
            elif msg == f"save_stereo_images":
                print(f'save stereo images...')
                save_stereo_images_flag = True
            elif msg == f"stop_save_stereo_images":
                print(f'stop save stereo images...')
                save_stereo_images_flag = False
        cam0.StopGrabbing()
        cam1.StopGrabbing()
        cam0.Close()
        cam1.Close()
        self.out_pipe.close()
        self.in_pipe.close()
        cv2.destroyAllWindows()
        print(f'cameras closed!')