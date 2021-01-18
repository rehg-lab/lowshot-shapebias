import torchvision.transforms as transforms
import numpy as np
import torch
import PIL

### modelnet image augmentation
class ModelNetImageAug(object):
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(brightness=(0.65,1.0),
                                        contrast=(0.7,1),
                                        saturation=(0,1),
                                        hue=(-0.5, 0.5))

        self.random_affine = transforms.RandomAffine(degrees=(-45.0,45.0),
                                         translate=(0.25,0.25),
                                         scale=(0.6, 1.25),
                                         shear=None,
                                         resample=PIL.Image.BILINEAR,
                                         fillcolor=(255,255,255))

        self.random_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.totensor = transforms.ToTensor()

    def apply_random_affine(self, img):
        t_img = self.random_affine(img)
        t = self.totensor(t_img)
        t[t==0]=1
        t_img = transforms.ToPILImage(mode='RGB')(t)

        return t_img


    def __call__(self, img):
        img = self.random_flip(img)
        img = self.apply_random_affine(img)
        img = self.color_jitter(img)

        return img

### point cloud transforms
### from:https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/data/data_utils.py

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()

class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points

class PointcloudSO3Rotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle_1 = np.random.uniform() * 2 * np.pi
        rotation_angle_2 = np.random.uniform() * 2 * np.pi
        rotation_angle_3 = np.random.uniform() * 2 * np.pi

        rotation_matrix_1 = angle_axis(rotation_angle_1, np.array([1.0,0.0,0.0]))
        rotation_matrix_2 = angle_axis(rotation_angle_2, np.array([0.0,1.0,0.0]))
        rotation_matrix_3 = angle_axis(rotation_angle_3, np.array([0.0,0.0,1.0]))

        normals = points.size(1) > 3
        if not normals:
            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_1.t())
            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_2.t())
            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_3.t())

            return points
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]

            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_1.t())
            points[:, 3:] = torch.matmul(points[:,3:], rotation_matrix_1.t())

            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_2.t())
            points[:, 3:] = torch.matmul(points[:,3:], rotation_matrix_2.t())

            points[:, 0:3] = torch.matmul(points[:,0:3], rotation_matrix_3.t())
            points[:, 3:] = torch.matmul(points[:,3:], rotation_matrix_3.t())

            return points


class PointcloudRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

