"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/solver.py
- Contains source code for the solver for training of the Digits-Five experiments
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from network.network_digits import GeneratorDigits, LabelClassifierDigits


class SolverDigits:
    def __init__(self, args, target, source, target_train_dataset, target_train_dataloader, target_test_dataloader,
                 source_train_dataloader, source_test_dataloader):
        """
        Initiate solver
        :param args: input arguments
        :param target: the name of the target dataset
        :param source: the array with the names of the source datasets
        :param target_train_dataset: target dataset
        :param target_train_dataloader: target training dataloader
        :param target_test_dataloader: target test dataloder
        :param source_train_dataloader: source train dataloader
        :param source_test_dataloader: source test dataloader
        """

        # basic settings
        self.target = target
        self.source = source
        self.num_classes = 10
        self.input_size = args.input_size
        self.ensemble_num = args.ensemble_num
        self.num_iter = args.source_data_num // args.batch_size
        self.iter_cnt = 0
        self.conf_threshold = args.conf_threshold  # for determining pseudo-labels
        self.pseudolabel_setting_interval = args.pseudolabel_setting_interval

        # dataloader setting
        self.t_train_dataset = target_train_dataset
        self.t_train_dataloader = target_train_dataloader
        self.s1_train_dataloader = source_train_dataloader[0]
        self.s2_train_dataloader = source_train_dataloader[1]
        self.s3_train_dataloader = source_train_dataloader[2]
        self.s4_train_dataloader = source_train_dataloader[3]

        self.t_test_dataloader = target_test_dataloader
        self.s1_test_dataloader = source_test_dataloader[0]
        self.s2_test_dataloader = source_test_dataloader[1]
        self.s3_test_dataloader = source_test_dataloader[2]
        self.s4_test_dataloader = source_test_dataloader[3]

        self.t_train_dataloader_iter = iter(self.t_train_dataloader)
        self.s1_train_dataloader_iter = iter(self.s1_train_dataloader)
        self.s2_train_dataloader_iter = iter(self.s2_train_dataloader)
        self.s3_train_dataloader_iter = iter(self.s3_train_dataloader)
        self.s4_train_dataloader_iter = iter(self.s4_train_dataloader)

        # device setting
        self.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

        # model setting
        self.G1 = GeneratorDigits().to(self.device)
        self.G2 = GeneratorDigits().to(self.device)
        self.LC1 = LabelClassifierDigits().to(self.device)
        self.LC2 = LabelClassifierDigits().to(self.device)
        self.LC_total = LabelClassifierDigits(input_size=2048*2).to(self.device)

        # training setting
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.decay = args.decay
        self.g_loss_weight = args.g_loss_weight
        self.lc_loss_weight = args.lc_loss_weight

        # optimizer
        self.g1_optimizer = optim.Adam(self.G1.parameters(), self.learning_rate, weight_decay=self.decay)
        self.g2_optimizer = optim.Adam(self.G2.parameters(), self.learning_rate, weight_decay=self.decay)
        self.lc1_optimizer = optim.Adam(self.LC1.parameters(), self.learning_rate, weight_decay=self.decay)
        self.lc2_optimizer = optim.Adam(self.LC2.parameters(), self.learning_rate, weight_decay=self.decay)
        self.lc_tot_optimizer = optim.Adam(self.LC_total.parameters(), self.learning_rate, weight_decay=self.decay)

        # lr scheduler
        self.g1_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.g1_optimizer, gamma=0.96)
        self.g2_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.g2_optimizer, gamma=0.96)
        self.lc1_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.lc1_optimizer, gamma=0.96)
        self.lc2_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.lc2_optimizer, gamma=0.96)
        self.lc_tot_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.lc_tot_optimizer, gamma=0.96)

    def get_train_samples(self):
        """ Return training samples from the target and source datasets """
        def get_sample(dataloader, dataloader_iter):
            try:
                sample = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                sample = next(dataloader_iter)
            return sample

        t_sample = get_sample(self.t_train_dataloader, self.t_train_dataloader_iter)
        s1_sample = get_sample(self.s1_train_dataloader, self.s1_train_dataloader_iter)
        s2_sample = get_sample(self.s2_train_dataloader, self.s2_train_dataloader_iter)
        s3_sample = get_sample(self.s3_train_dataloader, self.s3_train_dataloader_iter)
        s4_sample = get_sample(self.s4_train_dataloader, self.s4_train_dataloader_iter)
        return t_sample, s1_sample, s2_sample, s3_sample, s4_sample

    def get_mm_loss(self, t_feat, s1_feat, s2_feat, s3_feat, s4_feat, order):
        """
        Return the moment matching loss of the given order
        :param t_feat: target features
        :param s1_feat: source features 1
        :param s2_feat: source features 2
        :param s3_feat: source features 3
        :param s4_feat: source features 4
        :param order: order of the moment
        :return: moment matching loss in torch.tensor type
        """
        def get_norm(x, y):
            try:
                ans = torch.norm(x - y)
            except TypeError:
                ans = 0
            return ans

        def get_mean(x, k):
            if x is not None and x.size(0) > 0:
                return torch.mean(x ** k, dim=0)
            return None

        num = sum([feat is not None for feat in [t_feat, s1_feat, s2_feat, s3_feat, s4_feat]])
        if num <= 1:
            return torch.tensor(0).to(self.device)
        t_feat_mean = get_mean(t_feat, order)
        s1_feat_mean = get_mean(s1_feat, order)
        s2_feat_mean = get_mean(s2_feat, order)
        s3_feat_mean = get_mean(s3_feat, order)
        s4_feat_mean = get_mean(s4_feat, order)
        loss = get_norm(t_feat_mean, s1_feat_mean) + get_norm(t_feat_mean, s2_feat_mean) + \
               get_norm(t_feat_mean, s3_feat_mean) + get_norm(t_feat_mean, s4_feat_mean) + \
               get_norm(s1_feat_mean, s2_feat_mean) + get_norm(s1_feat_mean, s3_feat_mean) + \
               get_norm(s1_feat_mean, s4_feat_mean) + get_norm(s2_feat_mean, s3_feat_mean) + \
               get_norm(s2_feat_mean, s4_feat_mean) + get_norm(s3_feat_mean, s4_feat_mean)
        return loss

    def set_pseudolabels(self):
        # set dataloader to evaluate pseudolabels of every data in the target training dataset
        t_train_dataloader = DataLoader(self.t_train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)
        softmax = nn.Softmax(dim=-1).to(self.device)

        if self.ensemble_num == 1:  # no ensemble
            for sample in t_train_dataloader:
                image = sample['image'].to(self.device)
                indices = sample['index']
                num_image = image.size(0)
                if image.size(0) < self.batch_size:
                    image = torch.cat([image, torch.zeros([self.batch_size - num_image, image.size(1), image.size(2), image.size(3)], device=self.device)], dim=0)
                feat = self.G1(image)
                logits = self.LC1(feat)[:num_image]
                prob = softmax(logits)
                confidence, prediction = prob.max(dim=-1)

                accept = torch.where(confidence > self.conf_threshold, torch.tensor(1, device=self.device),
                                     torch.tensor(-1, device=self.device))
                prediction = prediction + 1
                ans = torch.clamp(prediction * accept, min=0).long() - 1
                self.t_train_dataset.labels[indices] = ans.cpu()
        elif self.ensemble_num == 2:  # ensemble of two network pairs
            for sample in t_train_dataloader:
                image = sample['image'].to(self.device)
                indices = sample['index']
                num_image = image.size(0)
                if image.size(0) < self.batch_size:
                    image = torch.cat([image, torch.zeros([self.batch_size - num_image, image.size(1), image.size(2), image.size(3)], device=self.device)], dim=0)
                feat1 = self.G1(image)
                feat2 = self.G2(image)
                feat = torch.cat([feat1, feat2], dim=1)
                logits = self.LC_total(feat)[:num_image]
                prob = softmax(logits)
                confidence, prediction = prob.max(dim=-1)

                accept = torch.where(confidence > self.conf_threshold, torch.tensor(1, device=self.device),
                                     torch.tensor(-1, device=self.device))
                prediction = prediction + 1
                ans = torch.clamp(prediction * accept, min=0).long() - 1
                self.t_train_dataset.labels[indices] = ans.cpu()

    def forward_G(self, t_img, s1_img, s2_img, s3_img, s4_img, index=1):
        """
        Return the target and source features using the feature extractor indicated as index
        :param t_img: target image
        :param s1_img: source image 1
        :param s2_img: source image 2
        :param s3_img: source image 3
        :param s4_img: source image 4
        :param index: index of the feature extractor
        :return: target and source features
        """
        if index == 1:
            generator = self.G1
        else:
            generator = self.G2
        t_feat = generator(t_img)
        s1_feat = generator(s1_img)
        s2_feat = generator(s2_img)
        s3_feat = generator(s3_img)
        s4_feat = generator(s4_img)
        return t_feat, s1_feat, s2_feat, s3_feat, s4_feat

    def forward_LC(self, s1_feat, s2_feat, s3_feat, s4_feat, index=1):
        """
        Return the prediction for the source features using the label classifier indicated as index
        :param s1_feat: source feature 1
        :param s2_feat: source feature 2
        :param s3_feat: source feature 3
        :param s4_feat: source feature 4
        :param index: index of the label classifier
        :return: source predictions
        """
        if index == 1:
            label_classifier = self.LC1
        elif index == 2:
            label_classifier = self.LC2
        else:
            label_classifier = self.LC_total
        s1_pred = label_classifier(s1_feat)
        s2_pred = label_classifier(s2_feat)
        s3_pred = label_classifier(s3_feat)
        s4_pred = label_classifier(s4_feat)
        return s1_pred, s2_pred, s3_pred, s4_pred

    def loss_mm(self, t_feat, s1_feat, s2_feat, s3_feat, s4_feat, t_label, s1_label, s2_label, s3_label, s4_label):
        """ Return 1st and 2nd order label-wise moment matching losses """
        loss_mm_1, loss_mm_2 = 0, 0
        for label in range(self.num_classes):
            t_indices = torch.where(t_label == label)
            s1_indices = torch.where(s1_label == label)
            s2_indices = torch.where(s2_label == label)
            s3_indices = torch.where(s3_label == label)
            s4_indices = torch.where(s4_label == label)
            t_feat_label = t_feat[t_indices]
            s1_feat_label = s1_feat[s1_indices]
            s2_feat_label = s2_feat[s2_indices]
            s3_feat_label = s3_feat[s3_indices]
            s4_feat_label = s4_feat[s4_indices]
            loss_mm_1 += self.get_mm_loss(t_feat_label, s1_feat_label, s2_feat_label, s3_feat_label, s4_feat_label, 1)
            loss_mm_2 += self.get_mm_loss(t_feat_label, s1_feat_label, s2_feat_label, s3_feat_label, s4_feat_label, 2)
        loss_mm_1 = loss_mm_1 / 10
        loss_mm_2 = loss_mm_2 / 10
        return loss_mm_1, loss_mm_2

    @staticmethod
    def loss_ce(s1_pred, s2_pred, s3_pred, s4_pred, s1_label, s2_label, s3_label, s4_label):
        """ Return label classification loss """
        criterion = nn.CrossEntropyLoss()
        loss_ce_1 = criterion(s1_pred, s1_label)
        loss_ce_2 = criterion(s2_pred, s2_label)
        loss_ce_3 = criterion(s3_pred, s3_label)
        loss_ce_4 = criterion(s4_pred, s4_label)
        loss_ce = (loss_ce_1 + loss_ce_2 + loss_ce_3 + loss_ce_4) / 4
        return loss_ce

    def train_mode_(self, *networks):
        for network in networks:
            network.train()

    def eval_mode_(self, *networks):
        for network in networks:
            network.eval()

    def train(self, epoch):
        """ Manage the overall training """

        print('\n*** Start epoch {:03d} ***'.format(epoch))

        self.train_mode_(self.G1, self.G2, self.LC1, self.LC2, self.LC_total)

        if self.ensemble_num == 1:
            self.MultiEPL_1_train(epoch)
        elif self.ensemble_num == 2:
            self.MultiEPL_2_train(epoch)

        self.g1_scheduler.step()
        self.g2_scheduler.step()
        self.lc1_scheduler.step()
        self.lc2_scheduler.step()
        self.lc_tot_scheduler.step()

        self.eval_mode_(self.G1, self.G2, self.LC1, self.LC2, self.LC_total)

    def MultiEPL_1_train(self, epoch):
        """ Training the model only with the label-wise moment matching """

        print('Epoch {:03d} --- MultiEPL-1 Start'.format(epoch))
        since = time.time()
        loss_mm_1, loss_mm_2, loss_ce = torch.tensor(0), torch.tensor(0), torch.tensor(0)

        for step in range(self.num_iter):
            # set target and sample images and labels
            t_sample, s1_sample, s2_sample, s3_sample, s4_sample = self.get_train_samples()

            t_image = t_sample['image'].to(self.device)
            s1_image = s1_sample['image'].to(self.device)
            s2_image = s2_sample['image'].to(self.device)
            s3_image = s3_sample['image'].to(self.device)
            s4_image = s4_sample['image'].to(self.device)

            t_label = t_sample['label'].to(self.device)  # pseudo-labels
            s1_label = s1_sample['label'].to(self.device)
            s2_label = s2_sample['label'].to(self.device)
            s3_label = s3_sample['label'].to(self.device)
            s4_label = s4_sample['label'].to(self.device)

            # get target and source features and source predictions
            t_feat, s1_feat, s2_feat, s3_feat, s4_feat = self.forward_G(t_image, s1_image, s2_image, s3_image, s4_image, 1)
            s1_pred, s2_pred, s3_pred, s4_pred = self.forward_LC(s1_feat, s2_feat, s3_feat, s4_feat, 1)

            # evaluate moment matching and label classification loss
            loss_mm_1, loss_mm_2 = self.loss_mm(t_feat, s1_feat, s2_feat, s3_feat, s4_feat, t_label, s1_label, s2_label, s3_label, s4_label)
            loss_ce = self.loss_ce(s1_pred, s2_pred, s3_pred, s4_pred, s1_label, s2_label, s3_label, s4_label)
            loss = (loss_mm_1 + loss_mm_2) * self.g_loss_weight + loss_ce * self.lc_loss_weight

            # model update
            self.G1.zero_grad()
            self.LC1.zero_grad()
            loss.backward()
            self.g1_optimizer.step()
            self.lc1_optimizer.step()

            # set pseudolabels every pseudolabel_setting_interval iterations
            self.iter_cnt += 1
            if self.iter_cnt % self.pseudolabel_setting_interval == self.pseudolabel_setting_interval - 1:
                self.set_pseudolabels()

        duration = int(time.time() - since)
        print('Epoch {:03d} --- Done, loss mm 1: {:06f}, loss mm 2: {:06f}, loss ce: {:06f}, duration: {:02d}m{:02d}s'
              .format(epoch, loss_mm_1.item(), loss_mm_2.item(), loss_ce.item(), duration // 60, duration % 60))

    def MultiEPL_2_train(self, epoch):
        """ Training the model with the label-wise moment matching and ensemble learning """
        print('Epoch {:03d} --- MultiEPL-2 Start'.format(epoch))
        since = time.time()
        loss_mm_1, loss_mm_2, loss_ce, loss_extractor_cls = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        for step in range(self.num_iter):
            # set target and sample images and labels
            t_sample, s1_sample, s2_sample, s3_sample, s4_sample = self.get_train_samples()

            t_image = t_sample['image'].to(self.device)
            s1_image = s1_sample['image'].to(self.device)
            s2_image = s2_sample['image'].to(self.device)
            s3_image = s3_sample['image'].to(self.device)
            s4_image = s4_sample['image'].to(self.device)

            t_label = t_sample['label'].to(self.device)  # pseudo-labels
            s1_label = s1_sample['label'].to(self.device)
            s2_label = s2_sample['label'].to(self.device)
            s3_label = s3_sample['label'].to(self.device)
            s4_label = s4_sample['label'].to(self.device)

            # get target and source features
            t_feat_1, s1_feat_1, s2_feat_1, s3_feat_1, s4_feat_1 = self.forward_G(t_image, s1_image, s2_image, s3_image, s4_image, 1)
            t_feat_2, s1_feat_2, s2_feat_2, s3_feat_2, s4_feat_2 = self.forward_G(t_image, s1_image, s2_image, s3_image, s4_image, 2)
            s1_feat_tot = torch.cat([s1_feat_1, s1_feat_2], dim=1).detach()
            s2_feat_tot = torch.cat([s2_feat_1, s2_feat_2], dim=1).detach()
            s3_feat_tot = torch.cat([s3_feat_1, s3_feat_2], dim=1).detach()
            s4_feat_tot = torch.cat([s4_feat_1, s4_feat_2], dim=1).detach()

            # get source predictions
            s1_pred_1, s2_pred_1, s3_pred_1, s4_pred_1 = self.forward_LC(s1_feat_1, s2_feat_1, s3_feat_1, s4_feat_1, 1)
            s1_pred_2, s2_pred_2, s3_pred_2, s4_pred_2 = self.forward_LC(s1_feat_2, s2_feat_2, s3_feat_2, s4_feat_2, 2)
            s1_pred_tot, s2_pred_tot, s3_pred_tot, s4_pred_tot = self.forward_LC(s1_feat_tot, s2_feat_tot, s3_feat_tot, s4_feat_tot, -1)

            # evaluate moment matching and label classification loss
            loss_mm_1_1, loss_mm_2_1 = self.loss_mm(t_feat_1, s1_feat_1, s2_feat_1, s3_feat_1, s4_feat_1, t_label, s1_label, s2_label, s3_label, s4_label)
            loss_mm_1_2, loss_mm_2_2 = self.loss_mm(t_feat_2, s1_feat_2, s2_feat_2, s3_feat_2, s4_feat_2, t_label, s1_label, s2_label, s3_label, s4_label)
            loss_ce_1 = self.loss_ce(s1_pred_1, s2_pred_1, s3_pred_1, s4_pred_1, s1_label, s2_label, s3_label, s4_label)
            loss_ce_2 = self.loss_ce(s1_pred_2, s2_pred_2, s3_pred_2, s4_pred_2, s1_label, s2_label, s3_label, s4_label)
            loss_ce_tot = self.loss_ce(s1_pred_tot, s2_pred_tot, s3_pred_tot, s4_pred_tot, s1_label, s2_label, s3_label, s4_label)

            loss_mm_1 = loss_mm_1_1 + loss_mm_1_2
            loss_mm_2 = loss_mm_2_1 + loss_mm_2_2
            loss_ce = loss_ce_1 + loss_ce_2 + loss_ce_tot
            loss = (loss_mm_1 + loss_mm_2) * self.g_loss_weight + loss_ce * self.lc_loss_weight

            # model update
            self.G1.zero_grad()
            self.G2.zero_grad()
            self.LC1.zero_grad()
            self.LC2.zero_grad()
            self.LC_total.zero_grad()
            loss.backward()
            self.g1_optimizer.step()
            self.g2_optimizer.step()
            self.lc1_optimizer.step()
            self.lc2_optimizer.step()
            self.lc_tot_optimizer.step()

            self.iter_cnt += 1
            if self.iter_cnt % self.pseudolabel_setting_interval == self.pseudolabel_setting_interval - 1:
                self.set_pseudolabels()

        duration = int(time.time() - since)
        print('Epoch {:03d} --- Done, loss mm 1: {:06f}, loss mm 2: {:06f}, loss ce: {:06f}, duration: {:02d}m{:02d}s'
              .format(epoch, loss_mm_1.item(), loss_mm_2.item(), loss_ce.item(), duration // 60, duration % 60))

    def test(self):
        correct = 0
        num_data = 0
        losses = 0
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.ensemble_num == 1:  # without ensemble
            for i, sample in enumerate(self.t_test_dataloader):
                image = sample['image'].to(self.device)
                label = sample['label'].to(self.device)

                feat = self.G1(image)
                logits = self.LC1(feat)
                loss = criterion(logits, label)
                pred = torch.argmax(logits, dim=-1)

                correct_check = (label == pred)
                correct += correct_check.sum().item()
                losses += loss.item()
                num_data += image.size(0)

        elif self.ensemble_num == 2:  # with ensemble of two network pairs
            for i, sample in enumerate(self.t_test_dataloader):
                image = sample['image'].to(self.device)
                label = sample['label'].to(self.device)

                feat1 = self.G1(image)
                feat2 = self.G2(image)
                feat = torch.cat([feat1, feat2], dim=1)
                logits = self.LC_total(feat)
                loss = criterion(logits, label)
                pred = torch.argmax(logits, dim=-1)

                correct_check = (label == pred)
                correct += correct_check.sum().item()
                losses += loss.item()
                num_data += image.size(0)

        acc = 100 * correct / num_data
        loss = losses / num_data
        return acc, loss
