
# flake8: noqa: F841

def test_skeleton_objects():
    from hipose.skeleton import Skeleton, SkeletonXsens, \
        SkeletonErgowear, SkeletonMTwAwinda, SkeletonXsensUpper

    s1 = Skeleton(root_joint=0, joint_names=["j1", "j2", "j3"],
                  segment_starts=[0, 1], segment_ends=[1, 2])
    s1 = SkeletonXsens()
    s2 = SkeletonMTwAwinda()
    s3 = SkeletonErgowear()
    s4 = SkeletonXsensUpper()


def test_skeleton_mappings():
    # TODO: add
    pass


def test_skeleton_kinematics():
    # TODO: add
    pass

def test_skeleton_vizualization():
    # TODO: add
    pass
