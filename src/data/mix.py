import torch
import numpy as np
import torch.nn as nn


class Mix(nn.Module):
    def __init__(
        self,
        p=1.0,
        alpha=0.4,
        additive=False,
        num_classes=1,
        num_classes_aux=1,
        same_class=False,
        mask_target=False,
        use_3d=False,
    ):
        """
        Initialize the Mixup augmentation module.

        Args:
            p (float, optional): Mixup probability. Defaults to 1.0.
            alpha (float, optional): Mixup interpolation parameter. Defaults to 0.4.
            additive (bool, optional): Whether to use additive mixup. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of auxiliary classes. Defaults to 1.
            same_class (bool, optional): Only mix samples of the same class. Defaults to False.
            mask_target (bool, optional): Whether to use mask target. Defaults to False.
        """
        super().__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.p = p
        self.same_class = same_class
        self.mask_target = mask_target
        self.use_3d = use_3d

    def get_permutation(self, y):
        """
        Creates a permutation of the data for mixing.

        Args:
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Permutation of the data.
        """
        # Create permutation for mixing
        if self.same_class and y is not None:
            # For same-class mixing, we need to create permutations within each class
            perm = torch.arange(y.shape[0]).to(y.device)

            # Convert y to class indices if it's one-hot encoded
            if len(y.shape) > 1 and y.shape[1] > 1:
                # One-hot encoded labels
                class_indices = y.argmax(dim=1)
            else:
                # Single class labels
                class_indices = y.long()

            # Create permutations within each class
            for class_idx in torch.unique(class_indices):
                class_mask = class_indices == class_idx
                class_indices_list = torch.where(class_mask)[0]
                if len(class_indices_list) > 1:
                    # Shuffle within this class
                    class_perm = class_indices_list[torch.randperm(len(class_indices_list))]
                    perm[class_indices_list] = class_perm

            assert (y[perm] == y).all()
        else:
            # Standard random permutation
            perm = torch.randperm(y.shape[0]).to(y.device)
        return perm

    def mix_y(self, y, perm, coeffs, num_classes):
        """
        Mixes the target labels y.

        Args:
            y (torch.Tensor [BS] or [BS, N_CLASSES]): Target labels.
            perm (torch.Tensor [BS]): Permutation of the data.
            coeffs (torch.Tensor [BS]): Coefficients for the mixup.
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor [BS, N_CLASSES]: Mixed target labels.
        """
        if num_classes > y.size(-1) or y.ndim == 1:  # One-hot
            y = torch.clamp(y, 0, num_classes - 1)
            y = (
                torch.zeros(y.size(0), num_classes)
                .to(y.device)
                .scatter(1, y.view(-1, 1).long(), 1)
            )

        if self.additive:
            y = torch.max(y, y[perm])
        else:
            if y.ndim == 1 or coeffs.ndim < 1:
                y = coeffs * y + (1 - coeffs) * y[perm]
            else:
                y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]
        return y

    def mix_mask(self, mask, perm, coeffs, num_classes):
        """
        Mixing for mask target.
        """
        raise NotImplementedError

    def mix_x(self, x, perm, coeffs):
        """
        Mixes the input data x with a random permutation.

        Args:
            x (torch.Tensor): Input data.
            perm (torch.Tensor): Permutation of the data.
            coeffs (torch.Tensor): Coefficients for the mixup.

        Returns:
            torch.Tensor: Mixed input data.
        """
        raise NotImplementedError

    def mix_track(self, imgs, coords, masks, targets, weak_pos=None):
        """
        Mixes tracking windows, keeping the union of their cells.

        The image is mixed by the subclass's mix_x, but the node sets are
        concatenated rather than mixed: a coordinate has no meaningful
        interpolation, and every cell of both windows really is present in the
        mixed image. Node padding therefore doubles, M -> 2M.

        Edge targets become block-diagonal — a cell can only be linked to a cell
        of the window it came from — so the tracker learns to follow one cell
        through a field of distractors that move independently of it.

        Args:
            imgs (torch.Tensor [BS, W, Z, Y, X]): Window images.
            coords (torch.Tensor [BS, W, M, 3]): Padded GT cell coordinates.
            masks (torch.Tensor [BS, W, M]): Which coords are real cells.
            targets (torch.Tensor [BS, W - 1, M, M]): GT transition matrices.
            weak_pos (torch.Tensor [BS, W, N, 3], optional): Weak positives,
                -1 padded. Defaults to None.

        Returns:
            tuple: (imgs, coords, masks, targets, weak_pos), node dims doubled.
        """
        bs = imgs.shape[0]
        perm, coeffs = self.get_perm_and_coefs(imgs)

        # Samples the permutation happens to leave in place (either p skipped
        # them or the shuffle was a fixed point) must not have their node set
        # doubled: a union with itself would list every cell twice, and the edge
        # target would then hold two equally-correct parents for each child.
        mixed = (perm != torch.arange(bs, device=perm.device)).view(bs, 1, 1)

        imgs = self._mix_x(imgs, perm, coeffs)
        coords = torch.cat([coords, coords[perm]], dim=2)
        masks = torch.cat([masks, masks[perm] & mixed], dim=2)

        zeros = torch.zeros_like(targets)
        other = targets[perm] * mixed.unsqueeze(-1)
        targets = torch.cat(
            [
                torch.cat([targets, zeros], dim=-1),
                torch.cat([zeros, other], dim=-1),
            ],
            dim=-2,
        )

        if weak_pos is not None:
            weak_pos = torch.cat(
                [
                    weak_pos,
                    torch.where(
                        mixed.unsqueeze(-1),
                        weak_pos[perm],
                        torch.full_like(weak_pos, -1.0),
                    ),
                ],
                dim=2,
            )
        return imgs, coords, masks, targets, weak_pos

    def mix_batch(self, batch, perm=None, coeffs=None):
        """
        Mixes a batch dict of dense tensors that all share the same spatial dims.

        Every sample is mixed with the same permutation and coefficients, so the
        image and its targets stay aligned.

        Args:
            batch (dict): name -> tensor [BS, ...]. Non-tensor values pass through.
            perm (torch.Tensor, optional): Pre-computed permutation. Defaults to None.
            coeffs (torch.Tensor, optional): Pre-computed coefficients. Defaults to None.

        Returns:
            dict: Batch with every tensor mixed.
        """
        raise NotImplementedError

    def reference_tensor(self, batch):
        """
        Picks the tensor that defines the batch size, device and spatial dims.

        Args:
            batch (dict): name -> tensor [BS, ...].

        Returns:
            torch.Tensor: The reference tensor.
        """
        if "image" in batch:
            return batch["image"]
        for value in batch.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("Batch contains no tensor to mix")

    def forward(self, x, y=None, y_aux=None, w=None):
        """
        Forward pass of the Mix module.

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor, optional): Target labels. Defaults to None.
            y_aux (torch.Tensor, optional): Auxiliary target labels. Defaults to None.
            w (torch.Tensor, optional): Sample weights. Defaults to None.

        Returns:
            tuple: Mixed data and labels.
        """
        raise NotImplementedError


class Mixup(Mix):
    """
    Mixup augmentation module.
    """
    def _mix_x(self, x, perm, coeffs):
        """
        Mixes the input data x with a random permutation of the same class.

        Args:
            x (torch.Tensor [BS, ...]): Input data.
            perm (torch.Tensor [BS]): Permutation of the data.
            coeffs (torch.Tensor [BS]): Coefficients for the mixup.

        Returns:
            torch.Tensor [BS, ...]: Mixed input data.
        """
        while coeffs.ndim < len(x.shape):
            coeffs = coeffs.unsqueeze(-1)

        return coeffs * x + (1 - coeffs) * x[perm]

    def mix_mask(self, mask, perm, coeffs, num_classes):
        """
        Mixing for mask target.

        Args:
            mask (torch.Tensor [BS, ...]): Mask.
            perm (torch.Tensor): Permutation of the data.
            coeffs (torch.Tensor): Coefficients for the mixup.
            num_classes (int): Number of classes.

        Returns:
            torch.Tensor: Mixed mask.
        """
        bs = mask.shape[0]

        if mask.size(1) != self.num_classes:
            if mask.size(1) == 1:
                mask = mask.squeeze(1)
            spatial_dims = mask.shape[1:]
            mask = mask.view(-1)
            mask = (
                torch.zeros(mask.size(0), num_classes)
                .to(mask.device)
                .scatter(1, mask.view(-1, 1).long(), 1)
            )
            mask = mask.reshape(bs, *spatial_dims, num_classes)
            mask = mask.permute(0, -1, *range(1, len(mask.shape) - 1))

        while coeffs.ndim < len(mask.shape):
            coeffs = coeffs.unsqueeze(-1)

        return coeffs * mask + (1 - coeffs) * mask[perm]

    def get_perm_and_coefs(self, y):
        """
        Get permutation and coefficients for mixup.

        Args:
            y (torch.Tensor): Target labels.

        Returns:
            tuple: A tuple containing (perm, coeffs) where perm is the permutation
                and coeffs are the mixup coefficients.
        """
        bs = y.shape[0]
        device = y.device

        perm = self.get_permutation(y)
        perm = torch.where(
            torch.rand(bs).to(device) < self.p,
            perm,
            torch.arange(bs).to(device),
        )
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(device)
        return perm, coeffs

    def mix_batch(self, batch, perm=None, coeffs=None):
        """
        Blends every dense tensor of the batch with the same per-sample coefficients.

        Boolean tensors (loss masks) cannot be blended, so a voxel is kept when it is
        set in either of the two mixed samples.

        Args:
            batch (dict): name -> tensor [BS, ...]. Non-tensor values pass through.
            perm (torch.Tensor, optional): Pre-computed permutation. Defaults to None.
            coeffs (torch.Tensor, optional): Pre-computed coefficients. Defaults to None.

        Returns:
            dict: Batch with every tensor mixed.
        """
        if self.p <= 0:
            return batch

        if (perm is None) or (coeffs is None):
            perm, coeffs = self.get_perm_and_coefs(self.reference_tensor(batch))

        mixed = {}
        for key, value in batch.items():
            if not torch.is_tensor(value):
                mixed[key] = value
            elif value.dtype == torch.bool:
                mixed[key] = value | value[perm]
            elif value.is_floating_point():
                mixed[key] = self._mix_x(value, perm, coeffs)
            else:  # integer targets (e.g. instance ids) cannot be blended
                mixed[key] = value
        return mixed

    def forward(self, x, y=None, y_aux=None, w=None, perm=None, coeffs=None):
        """
        Forward pass of the Mixup module.

        Called with a single batch dict, mixes it and returns the mixed dict.

        Args:
            x (torch.Tensor or dict): Input data. Can be a tensor or a dictionary of tensors.
            y (torch.Tensor): Target labels for the main task.
            y_aux (torch.Tensor, optional): Target labels for the auxiliary task. Defaults to None.
            w (torch.Tensor, optional): Sample weights. Defaults to None.
            perm (torch.Tensor, optional): Pre-computed permutation. Defaults to None.
            coeffs (torch.Tensor, optional): Pre-computed coefficients. Defaults to None.

        Returns:
            tuple: A tuple containing (x, y, y_aux, w) where all are mixed versions
                of the input data and labels.
            dict: The mixed batch, when called as `mix(batch)`.
        """
        if isinstance(x, dict) and y is None:
            return self.mix_batch(x, perm=perm, coeffs=coeffs)

        if self.p <= 0:
            return x, y, y_aux, w

        if (perm is None) and (coeffs is None):
            perm, coeffs = self.get_perm_and_coefs(y)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = self._mix_x(v, perm, coeffs)
        elif x is None:
            pass
        else:
            x = self._mix_x(x, perm, coeffs)

        if y is not None:
            if self.mask_target:
                y = self.mix_mask(y, perm, coeffs, self.num_classes)
            else:
                y = self.mix_y(y, perm, coeffs, self.num_classes)

        if y_aux is not None and self.num_classes_aux > 0:
            y_aux = self.mix_y(y_aux, perm, coeffs, self.num_classes_aux)

        if w is not None:
            w = coeffs * w + (1 - coeffs) * w[perm]

        return x, y, y_aux, w


class Cutmix(Mix):
    """
    Cutmix augmentation module.
    """
    @staticmethod
    def rand_bbox(size, lam):
        """
        Returns the coordinates of a random rectangle in the image for cutmix.

        Args:
            size (torch.Tensor): Input size [batch_size x c x w x h].
            lam (float): Lambda sampled by the beta distribution. Controls the size of the bbox.

        Returns:
            List[int]: 4 coordinates of the rectangle (bbx1, bby1, bbx2, bby2).
            float: Proportion of the unmasked image.
        """
        w, h = size[-2:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        prop = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return [bbx1, bby1, bbx2, bby2], prop

    @staticmethod
    def rand_bbox_3d(size, lam):
        """
        Returns the coordinates of a random cube in the image for cutmix.

        Args:
            size (torch.Tensor): Input size [batch_size x c x w x h].
            lam (float): Lambda sampled by the beta distribution. Controls the size of the bbox.

        Returns:
            List[int]: 6 coordinates of the cube (bbx1, bby1, bbz1, bbx2, bby2, bbz2).
            float: Proportion of the unmasked image.
        """
        w, h, d = size[-3:]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cut_d = int(d * cut_rat)

        # uniform. The centre spans the whole volume and the box is clipped to it:
        # sampling inside [cut/2, size - cut/2] has an empty range once lam -> 0.
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cz = np.random.randint(d)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbz1 = np.clip(cz - cut_d // 2, 0, d)

        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        bbz2 = np.clip(cz + cut_d // 2, 0, d)

        prop = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (w * h * d))
        return [bbx1, bby1, bbz1, bbx2, bby2, bbz2], prop

    def _mix_x(self, x, perm, bbox):
        """
        Mixes the input data x with a random permutation of the same class.

        Args:
            x (torch.Tensor [BS, ...]): Input data.
            perm (torch.Tensor [BS]): Permutation of the data.
            coeffs (torch.Tensor [BS]): Coefficients for the mixup.

        Returns:
            torch.Tensor [BS, ...]: Mixed input data.
        """
        if len(bbox) == 4:
            bbx1, bby1, bbx2, bby2 = bbox
            x[:, ..., bbx1:bbx2, bby1:bby2] = x[perm, ..., bbx1:bbx2, bby1:bby2]
        elif len(bbox) == 6:
            bbx1, bby1, bbz1, bbx2, bby2, bbz2 = bbox
            x[:, ..., bbx1:bbx2, bby1:bby2, bbz1:bbz2] = x[
                perm, ..., bbx1:bbx2, bby1:bby2, bbz1:bbz2
            ]
        else:
            raise NotImplementedError
        return x

    def get_perm_and_coefs(self, y):
        """
        Get permutation and coefficients for mixup.

        Args:
            y (torch.Tensor): Target labels.

        Returns:
            tuple: A tuple containing (perm, coeffs) where perm is the permutation
                and coeffs are the mixup coefficients.
        """
        bs = y.shape[0]
        device = y.device

        perm = self.get_permutation(y)
        perm = torch.where(
            torch.rand(bs).to(device) < self.p,
            perm,
            torch.arange(bs).to(device),
        )
        coeff = self.beta_distribution.rsample(torch.Size((1,))).view(-1).item()
        return perm, coeff

    def mix_batch(self, batch, perm=None, coeffs=None):
        """
        Pastes the same random box from the permuted samples into every batch tensor.

        Unlike mixup this is dtype agnostic: the region is copied, not blended, so
        boolean masks and integer targets stay valid.

        Args:
            batch (dict): name -> tensor [BS, ...]. Non-tensor values pass through.
            perm (torch.Tensor, optional): Pre-computed permutation. Defaults to None.
            coeffs (float, optional): Pre-computed coefficient. Defaults to None.

        Returns:
            dict: Batch with every tensor mixed.
        """
        if self.p <= 0:
            return batch

        reference = self.reference_tensor(batch)
        if (perm is None) or (coeffs is None):
            perm, coeffs = self.get_perm_and_coefs(reference)

        if self.use_3d:
            bbox, _ = self.rand_bbox_3d(reference.size(), coeffs)
        else:
            bbox, _ = self.rand_bbox(reference.size(), coeffs)

        # _mix_x writes in place, so copy to leave the incoming batch untouched
        return {
            key: self._mix_x(value.clone(), perm, bbox) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def forward(self, x, y=None, y_aux=None, w=None, perm=None, coeffs=None):
        """
        Forward pass of the Cutmix module.

        Called with a single batch dict, mixes it and returns the mixed dict.

        Args:
            x (torch.Tensor or dict): Input data. Can be a tensor or a dictionary of tensors.
            y (torch.Tensor): Target labels for the main task.
            y_aux (torch.Tensor, optional): Target labels for the auxiliary task. Defaults to None.
            w (torch.Tensor, optional): Sample weights. Defaults to None.
            perm (torch.Tensor, optional): Pre-computed permutation. Defaults to None.
            coeffs (torch.Tensor, optional): Pre-computed coefficients. Defaults to None.

        Returns:
            tuple: A tuple containing (x, y, y_aux, w) where all are mixed versions
                of the input data and labels.
            dict: The mixed batch, when called as `mix(batch)`.
        """
        if isinstance(x, dict) and y is None:
            return self.mix_batch(x, perm=perm, coeffs=coeffs)

        if self.p <= 0:
            return x, y, y_aux, w

        if (perm is None) and (coeffs is None):
            perm, coeff = self.get_perm_and_coefs(y)
        else:
            coeff = coeffs

        # Get bbox coords
        if self.use_3d:
            bbox, prop = self.rand_bbox_3d(
                x.size(), coeff
            )
        else:
            bbox, prop = self.rand_bbox(x.size(), coeff)

        if isinstance(x, dict):
            for k, v in x.items():
                x[k] = self._mix_x(v, perm, bbox)
        elif x is None:
            pass
        else:
            x = self._mix_x(x, perm, bbox)

        if y is not None:
            if self.mask_target:
                y = self._mix_x(y, perm, bbox)
            else:
                y = self.mix_y(y, perm, prop, self.num_classes)

        if y_aux is not None and self.num_classes_aux > 0:
            if self.use_3d:
                y_aux = self.mix_y(y_aux, perm, prop, self.num_classes_aux)
            else:
                ratio = x.size(-1) / y_aux.size(-1)
                bbx1, bby1, bbx2, bby2 = bbox[-4:]
                bbx1 = int(bbx1 / ratio)
                bby1 = int(bby1 / ratio)
                bbx2 = int(bbx2 / ratio) + 1
                bby2 = int(bby2 / ratio) + 1
                bbox_downscaled = [bbx1, bby1, bbx2, bby2]
                y_aux = self._mix_x(y_aux, perm, bbox_downscaled)

        if w is not None:
            w = prop * w + (1 - prop) * w[perm]

        return x, y, y_aux, w
