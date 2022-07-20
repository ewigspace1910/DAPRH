import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd

# torch.set_printoptions(threshold=np.inf)


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, inputs2, indexes, source_classes):
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        
        def softmax_mse_loss(input_logits, target_logits):
            """Takes softmax on both sides and returns MSE loss

            Note:
            - Returns the sum over all examples. Divide by the batch size afterwards
              if you want the mean.
            - Sends gradients to inputs but not the targets.
            """
            assert input_logits.size() == target_logits.size()
            input_softmax = F.softmax(input_logits, dim=1)
            target_softmax = F.softmax(target_logits, dim=1)
            num_classes = input_logits.size()[1]
            return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        B = inputs.size(0)
        N = inputs.size(1)
        C = labels.max() + 1

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())

        # mask_ema = (nums>=2).float().expand_as(sim)
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)

        if source_classes != 0:
            # mask_ema = torch.cat((torch.ones(source_classes),
            #                       torch.zeros(labels.max()+1-source_classes)), dim=0).long().cuda()
            # mask_ema = mask_ema.nonzero().squeeze()
            mask_ema = torch.arange(source_classes).cuda()
        else:
            mask_ema = (nums.squeeze() >= 2).long().nonzero().squeeze().cuda()

        inputs2 = inputs2.mm(self.features.t()) / self.temp
        sim2 = torch.zeros(labels.max()+1, B).float().cuda()
        sim2.index_add_(0, labels, inputs2.t().contiguous())
        sim2 /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        p1 = torch.index_select(sim, 0, mask_ema)
        p2 = torch.index_select(sim2, 0, mask_ema)
        loss_mse = softmax_mse_loss(p1.t().contiguous(), p2.t().contiguous())
        
        mask_hard = (nums > 0).float()
        mask_hard = mask_hard.expand(C, B).t().contiguous()
        _, indices = labels.expand(B, N).sort(1)
        result = torch.gather(inputs, dim=1, index=indices)
        patch = torch.split(result, nums.squeeze(dim=1).long().cpu().numpy().tolist(), dim=1)
        list = []
        for tensor in patch:
            if tensor.shape == torch.Size([B, 0]):
                list.append(torch.zeros(B, 1).cuda())
            else:
                x = tensor.max(1)[0].cuda()
                list.append(x.view(-1, 1))
        sim_hard = torch.cat(tuple(list), dim=1)

        for i in range(B):
            mask_idx = (labels == targets[i]).long()
            idx = (mask_idx != 0).nonzero()
            sim_hard[i][targets[i]] = inputs[i][idx].min()

        sim_hard = sim_hard.cuda()
        masked_sim = masked_softmax(sim_hard, mask_hard)
        loss_con = F.nll_loss(torch.log(masked_sim + 1e-6), targets)

        # mask = mask.expand_as(sim)
        # masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        # loss_con = F.nll_loss(torch.log(masked_sim+1e-6), targets)
        return loss_con, loss_mse


