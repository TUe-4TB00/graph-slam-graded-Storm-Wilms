
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

odom = gtsam.Pose2(2.0, 0.0, np.deg2rad(45.0))

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    graph.add(
        gtsam.BetweenFactorPose2(
            X(4),
            X(5),
            odom,
            ODOMETRY_NOISE
        )
    )
    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    pose_4 = initial_estimate.atPose2(X(4))
    pose_5 = pose_4.compose(odom)
    initial_estimate.insert(X(5), pose_5)

    return graph, initial_estimate