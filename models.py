import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MAMLFCNet(nn.Module):
    """
    Fully Connected Network for MAML based on the description in
    Finn et al. (2017) for Omniglot non-convolutional results.
    Includes 4 hidden layers with specified sizes and batch normalization.
    """

    def __init__(self, n_way, input_dim=784):  # input_dim=28*28 for Omniglot
        super().__init__()
        self.n_way = n_way
        self.input_dim = input_dim

        # Define hidden layer sizes as per paper description
        h_sizes = [256, 128, 64, 64]

        # Define layers: Linear -> BatchNorm -> ReLU
        self.fc1 = nn.Linear(input_dim, h_sizes[0])
        self.bn1 = nn.BatchNorm1d(h_sizes[0])  # BatchNorm for FC layers is 1D

        self.fc2 = nn.Linear(h_sizes[0], h_sizes[1])
        self.bn2 = nn.BatchNorm1d(h_sizes[1])

        self.fc3 = nn.Linear(h_sizes[1], h_sizes[2])
        self.bn3 = nn.BatchNorm1d(h_sizes[2])

        self.fc4 = nn.Linear(h_sizes[2], h_sizes[3])
        self.bn4 = nn.BatchNorm1d(h_sizes[3])

        # Final classifier layer
        self.classifier = nn.Linear(h_sizes[3], n_way)

        # Xavier initialization for linear layers (BatchNorm defaults are usually fine)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Default initialization for BatchNorm1d (affine=True):
            # weight ~ U(0,1), bias = 0 (older PyTorch) or weight=1, bias=0 (newer PyTorch)

    def forward(self, x, params=None):
        # x shape: [batch_size, channels, height, width] or [batch_size, features]
        batch_size = x.size(0)
        # Ensure input is flattened: maintain batch dim, flatten the rest
        x = x.view(batch_size, -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dim {self.input_dim}, but got {x.shape[1]}"
            )

        if params is None:
            # Use model's parameters (standard forward pass)
            # Layer 1: fc -> bn -> relu
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)

            # Layer 2: fc -> bn -> relu
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)

            # Layer 3: fc -> bn -> relu
            x = self.fc3(x)
            x = self.bn3(x)
            x = F.relu(x)

            # Layer 4: fc -> bn -> relu
            x = self.fc4(x)
            x = self.bn4(x)
            x = F.relu(x)

            # Classifier
            x = self.classifier(x)
            return x
        else:
            # Use provided functional parameters (for MAML inner loop)
            # Order of params must match get_parameters()!
            # [fc1.w, fc1.b, bn1.w, bn1.b, fc2.w, fc2.b, bn2.w, bn2.b, ...]

            # Layer 1: fc -> bn -> relu
            idx = 0
            x = F.linear(x, params[idx], params[idx + 1])  # fc1.w, fc1.b
            x = F.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=params[idx + 2],
                bias=params[idx + 3],
                training=True,
            )  # bn1.w, bn1.b
            x = F.relu(x)

            # Layer 2: fc -> bn -> relu
            idx = 4
            x = F.linear(x, params[idx], params[idx + 1])  # fc2.w, fc2.b
            x = F.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=params[idx + 2],
                bias=params[idx + 3],
                training=True,
            )  # bn2.w, bn2.b
            x = F.relu(x)

            # Layer 3: fc -> bn -> relu
            idx = 8
            x = F.linear(x, params[idx], params[idx + 1])  # fc3.w, fc3.b
            x = F.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=params[idx + 2],
                bias=params[idx + 3],
                training=True,
            )  # bn3.w, bn3.b
            x = F.relu(x)

            # Layer 4: fc -> bn -> relu
            idx = 12
            x = F.linear(x, params[idx], params[idx + 1])  # fc4.w, fc4.b
            x = F.batch_norm(
                x,
                running_mean=None,
                running_var=None,
                weight=params[idx + 2],
                bias=params[idx + 3],
                training=True,
            )  # bn4.w, bn4.b
            x = F.relu(x)

            # Classifier
            idx = 16
            x = F.linear(x, params[idx], params[idx + 1])  # classifier.w, classifier.b
            return x

    def get_parameters(self):
        """
        Return parameters in the order expected by the functional forward pass.
        Order: [fc1.w, fc1.b, bn1.w, bn1.b, fc2.w, fc2.b, bn2.w, bn2.b, ...]
        """
        return [
            self.fc1.weight,
            self.fc1.bias,
            self.bn1.weight,
            self.bn1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.bn2.weight,
            self.bn2.bias,
            self.fc3.weight,
            self.fc3.bias,
            self.bn3.weight,
            self.bn3.bias,
            self.fc4.weight,
            self.fc4.bias,
            self.bn4.weight,
            self.bn4.bias,
            self.classifier.weight,
            self.classifier.bias,
        ]


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    # Conv -> BatchNorm (momentum=1.0) -> ReLU -> MaxPool2d
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, momentum=1.0, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MAMLConvNet(nn.Module):
    def __init__(self, n_way, in_channels=3, hidden_size=32):
        """
        Four-layer convolutional network as in the MAML paper.
        """
        super(MAMLConvNet, self).__init__()
        self.n_way = n_way

        self.layer1 = conv_block(in_channels, hidden_size)
        self.layer2 = conv_block(hidden_size, hidden_size)
        self.layer3 = conv_block(hidden_size, hidden_size)
        self.layer4 = conv_block(hidden_size, hidden_size)
        # For 84x84 input, after 4 poolings we get ~5x5 feature maps.
        if in_channels == 3:
            self.classifier = nn.Linear(hidden_size * 5 * 5, n_way)
        else:
            self.classifier = nn.Linear(hidden_size * 1 * 1, n_way)

        # Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, params=None):
        if params is None:
            # nn.MaxPool2d handles indices internally when needed by autograd,
            # so this part is likely fine.
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)
        else:
            # Modify the functional part
            p = params
            # Layer 1
            out = F.conv2d(x, p[0], p[1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=None,
                running_var=None,
                weight=p[2],
                bias=p[3],
                training=True,
                momentum=0,
            )
            out = F.relu(out, inplace=True)
            # Get both output and indices, but only pass output forward
            # return_indices=True needed for torch.mps() backend
            out, _ = F.max_pool2d(out, kernel_size=2, return_indices=True)

            # Layer 2
            idx = 4
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=None,
                running_var=None,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=0,
            )
            out = F.relu(out, inplace=True)
            out, _ = F.max_pool2d(out, kernel_size=2, return_indices=True)

            # Layer 3
            idx = 8
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=None,
                running_var=None,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=0,
            )
            out = F.relu(out, inplace=True)
            out, _ = F.max_pool2d(out, kernel_size=2, return_indices=True)

            # Layer 4
            idx = 12
            out = F.conv2d(out, p[idx], p[idx + 1], stride=1, padding=1)
            out = F.batch_norm(
                out,
                running_mean=None,
                running_var=None,
                weight=p[idx + 2],
                bias=p[idx + 3],
                training=True,
                momentum=0,
            )
            out = F.relu(out, inplace=True)
            out, _ = F.max_pool2d(out, kernel_size=2, return_indices=True)

            out = out.view(out.size(0), -1)
            # Classifier layer
            idx = 16
            logits = F.linear(out, p[idx], p[idx + 1])
        return logits

    def get_parameters(self):
        # Return parameters as a list (order must match the forward() implementation above)
        params = []
        # Layer 1: conv, bn
        params += [
            self.layer1[0].weight,
            self.layer1[0].bias,
            self.layer1[1].weight,
            self.layer1[1].bias,
        ]
        # Layer 2: conv, bn
        params += [
            self.layer2[0].weight,
            self.layer2[0].bias,
            self.layer2[1].weight,
            self.layer2[1].bias,
        ]
        # Layer 3: conv, bn
        params += [
            self.layer3[0].weight,
            self.layer3[0].bias,
            self.layer3[1].weight,
            self.layer3[1].bias,
        ]
        # Layer 4: conv, bn
        params += [
            self.layer4[0].weight,
            self.layer4[0].bias,
            self.layer4[1].weight,
            self.layer4[1].bias,
        ]
        # Classifier
        params += [self.classifier.weight, self.classifier.bias]
        return params


#################################
# Meta Wrapper for MAML (Manual Inner-Loop Updates)
#################################


class Meta(nn.Module):
    def __init__(self, args, model):
        super(Meta, self).__init__()
        self.args = args
        self.net = model  # base learner

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        Training: perform inner-loop adaptation on the support set and compute query loss.
        """
        # Get initial fast weights directly from the network
        fast_weights = self.net.get_parameters()

        # Make sure all parameters require gradients
        for i, w in enumerate(fast_weights):
            if not w.requires_grad:
                print(f"Warning: Parameter {i} does not require grad")
                w.requires_grad = True

        # Inner-loop adaptation
        inner_losses = []
        inner_accs = []
        for _ in range(self.args.update_step):
            logits = self.net.forward(x_spt, params=fast_weights)
            loss = F.cross_entropy(logits, y_spt)
            inner_losses.append(loss.item())

            pred_spt = torch.argmax(logits, dim=1)
            correct_spt = torch.eq(pred_spt, y_spt).sum().item()
            acc_spt = correct_spt / float(len(y_spt))
            inner_accs.append(acc_spt)

            # Check if loss is valid
            # TODO: Remove since its never really used
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}")
                # Use a dummy loss to avoid breaking the training
                loss = torch.tensor(0.1, device=loss.device, requires_grad=True)

            # Create gradients
            grads = torch.autograd.grad(
                loss,
                fast_weights,
                create_graph=False if self.args.first_order else True,
                allow_unused=True,
            )

            # Check gradients
            # TODO: Remove since its never really used
            if any(g is None for g in grads):
                print("Warning: Some gradients are None")
                # Create dummy gradients where needed
                grads = [
                    g if g is not None else torch.zeros_like(w)
                    for g, w in zip(grads, fast_weights)
                ]

            # Update fast weights.
            if self.args.first_order:
                fast_weights = [
                    w - self.args.inner_lr * g.detach()
                    for w, g in zip(fast_weights, grads)
                ]
            else:
                fast_weights = [
                    w - self.args.inner_lr * g for w, g in zip(fast_weights, grads)
                ]

        # Evaluate on query set using adapted weights.

        avg_inner_loss = sum(inner_losses) / len(inner_losses)
        avg_inner_acc = sum(inner_accs) / len(inner_accs)

        logits_q = self.net.forward(x_qry, params=fast_weights)
        qry_loss = F.cross_entropy(logits_q, y_qry)

        # Calculate accuracy
        pred = torch.argmax(logits_q, dim=1)
        correct = torch.eq(pred, y_qry).sum().item()
        acc = correct / float(len(y_qry))
        return qry_loss, acc, avg_inner_loss, avg_inner_acc

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Finetuning for evaluation. Returns accuracies after each update step.
        This procedure is done without computing higher-order gradients.
        """
        net_copy = copy.deepcopy(self.net)
        fast_weights = net_copy.get_parameters()
        accs = []

        # Evaluate before any update.
        logits_q = net_copy.forward(x_qry, params=fast_weights)
        pred = torch.argmax(logits_q, dim=1)
        correct = torch.eq(pred, y_qry).sum().item()
        acc = correct / float(len(y_qry))
        accs.append(acc)

        for _ in range(self.args.update_step_test):
            logits = net_copy.forward(x_spt, params=fast_weights)
            loss = F.cross_entropy(logits, y_spt)

            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value during finetuning: {loss.item()}")
                loss = torch.tensor(0.1, device=loss.device, requires_grad=True)

            grads = torch.autograd.grad(loss, fast_weights)
            # Apply first-order option if specified
            if self.args.first_order:
                grads = [g.detach() for g in grads]

            fast_weights = [
                w - self.args.inner_lr * g for w, g in zip(fast_weights, grads)
            ]

            logits_q = net_copy.forward(x_qry, params=fast_weights)
            pred = torch.argmax(logits_q, dim=1)
            correct = torch.eq(pred, y_qry).sum().item()
            acc = correct / float(len(y_qry))
            accs.append(acc)

        return accs
