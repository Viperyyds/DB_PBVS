import cv2
import numpy as np


def draw_conner_box(show_img, corners, tag_id, color=[0, 255, 255]):
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # Convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    # Draw the bounding box of the ArUCo detection
    # cv2.line(show_img, topLeft, topRight, color, 1)
    # cv2.line(show_img, topRight, bottomRight, color, 1)
    # cv2.line(show_img, bottomRight, bottomLeft, color, 1)
    # cv2.line(show_img, bottomLeft, topLeft, color, 1)
    # Compute and draw the center (x, y) coordinates of the ArUCo marker
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    # cv2.circle(show_img, (cX, cY), 4, (0, 0, 255), -1)

    cv2.drawMarker(show_img, position=(cX, cY), color=(0, 0, 255), markerSize=5,
                   markerType=cv2.MARKER_CROSS, thickness=2)

    cv2.putText(show_img, str(1), (topLeft[0], topLeft[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    cv2.putText(show_img, str(2), (topRight[0], topRight[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)

    # cv2.drawMarker(show_img, position=topRight, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # cv2.drawMarker(show_img, position=bottomRight, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # cv2.drawMarker(show_img, position=bottomLeft, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # cv2.drawMarker(show_img, position=topLeft, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # pix = original_corners[i, :]
    show_img[int(topLeft[1]), int(topLeft[0]), :] = np.array([255, 0, 0])
    show_img[int(topRight[1]), int(topRight[0]), :] = np.array([255, 0, 0])
    show_img[int(bottomRight[1]), int(bottomRight[0]), :] = np.array([0, 0, 255])
    show_img[int(bottomLeft[1]), int(bottomLeft[0]), :] = np.array([0, 0, 255])
    # cv2.drawMarker(show_img, position=topLeft, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # Draw the ArUco marker ID on the image
    cv2.putText(show_img, str(tag_id), ((topLeft[0]+topRight[0])//2, (topLeft[1]+topRight[1])//2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)


def draw_box_corners(show_img, corners, tag_id, color=[0, 255, 255]):
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # Convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    # Draw the bounding box of the ArUCo detection
    # cv2.line(show_img, topLeft, topRight, color, 1)
    # cv2.line(show_img, topRight, bottomRight, color, 1)
    # cv2.line(show_img, bottomRight, bottomLeft, color, 1)
    # cv2.line(show_img, bottomLeft, topLeft, color, 1)
    # Compute and draw the center (x, y) coordinates of the ArUCo marker
    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    # cv2.circle(show_img, (cX, cY), 4, (0, 0, 255), -1)

    cv2.drawMarker(show_img, position=(cX, cY), color=(0, 0, 255), markerSize=5,
                   markerType=cv2.MARKER_CROSS, thickness=2)

    cv2.putText(show_img, str(1), (topLeft[0], topLeft[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    cv2.putText(show_img, str(2), (topRight[0], topRight[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    # cv2.drawMarker(show_img, position=topRight, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # cv2.drawMarker(show_img, position=bottomRight, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # cv2.drawMarker(show_img, position=bottomLeft, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # pix = original_corners[i, :]
    show_img[int(topLeft[1]), int(topLeft[0]), :] = np.array([255, 0, 0])
    show_img[int(topRight[1]), int(topRight[0]), :] = np.array([255, 0, 255])
    show_img[int(bottomRight[1]), int(bottomRight[0]), :] = np.array([0, 0, 255])
    show_img[int(bottomLeft[1]), int(bottomLeft[0]), :] = np.array([0, 0, 255])
    # cv2.drawMarker(show_img, position=topLeft, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    # Draw the ArUco marker ID on the image
    cv2.putText(show_img, str(tag_id), ((topLeft[0]+topRight[0])//2, (topLeft[1]+topRight[1])//2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)

def draw_box_corners1(show_img, corners, tag_id, color=[0, 255, 255]):
    (topLeft, topRight, bottomRight, bottomLeft) = corners
    # Convert each of the (x, y)-coordinate pairs to integers
    topRight = (int(topRight[0]), int(topRight[1]))
    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
    topLeft = (int(topLeft[0]), int(topLeft[1]))
    # Draw the bounding box of the ArUCo detection
    # cv2.line(show_img, topLeft, topRight, color, 2)
    # cv2.line(show_img, topRight, bottomRight, color, 2)
    # cv2.line(show_img, bottomRight, bottomLeft, color, 2)
    # cv2.line(show_img, bottomLeft, topLeft, color, 2)
    # Compute and draw the center (x, y) coordinates of the ArUCo marker
    # cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    # cY = int((topLeft[1] + bottomRight[1]) / 2.0)
    # cv2.circle(show_img, (cX, cY), 4, (0, 0, 255), -1)

    # cv2.drawMarker(show_img, position=(cX, cY), color=(0, 0, 255), markerSize=5,
    #                markerType=cv2.MARKER_CROSS, thickness=2)

    cv2.putText(show_img, str(1), (topLeft[0], topLeft[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    cv2.putText(show_img, str(2), (topRight[0], topRight[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    # cv2.drawMarker(show_img, position=topRight, color=(0, 255, 0), markerSize=15,
    #                markerType=cv2.MARKER_CROSS, thickness=3)
    cv2.drawMarker(show_img, position=bottomRight, color=(0, 255, 0), markerSize=3,
                   markerType=cv2.MARKER_CROSS, thickness=1)
    cv2.drawMarker(show_img, position=bottomLeft, color=(0, 255, 0), markerSize=3,
                   markerType=cv2.MARKER_CROSS, thickness=1)
    cv2.drawMarker(show_img, position=topLeft, color=(0, 255, 0), markerSize=3,
                   markerType=cv2.MARKER_CROSS, thickness=1)

    cv2.drawMarker(show_img, position=topRight, color=(0, 255, 0), markerSize=3,
                   markerType=cv2.MARKER_CROSS, thickness=1)
    # Draw the ArUco marker ID on the image
    cv2.putText(show_img, str(tag_id), ((topLeft[0]+topRight[0])//2, (topLeft[1]+topRight[1])//2 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)