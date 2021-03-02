import numpy as np
import numpy.linalg as la


class test:

    def __init__(self, data, feats, num_train=2):
        self.data = data
        self.U, self.sigma, self.V, self.w = feats
        self.acc = 0.0
        self.err = []
        self.num_train = num_train

    def get_err(self, w_test):
        err = []
        for l, wt in self.w:
            err.append((l, la.norm(wt - w_test)))
        return err

    def test(self):
        num_images = 0
        # print(self.data)
        for k, v in self.data.items():
            for i, img in enumerate(v):
                if i >= self.num_train:
                    num_images += 1
                    test_img = np.reshape(img, (-1, 1))
                    w_test = np.squeeze(self.U.transpose() @ test_img)
                    self.err = self.get_err(w_test)
                    pred_label = self.err[np.argmin([e for _, e in self.err])][0]
                    if pred_label == k:
                        self.acc += 1
        self.acc /= num_images
        return self.acc
