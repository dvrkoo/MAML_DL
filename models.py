import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MAMLFCNet(nn.Module):
    def __init__(self, n_way, input_dim=784, hidden_size=256):
        super().__init__()
        self.n_way = n_way
        self.input_dim = input_dim

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, n_way)

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, params=None):
        batch_size = x.size(0)
        # Correctly reshape: maintain batch dimension, flatten the rest
        x = x.view(batch_size, -1)

        if params is None:
            # Use model's parameters
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.classifier(x)
            return x
        else:
            # Use custom params
            x = F.linear(x, params[0], params[1])
            x = F.relu(x)
            x = F.linear(x, params[2], params[3])
            x = F.relu(x)
            x = F.linear(x, params[4], params[5])
            x = F.relu(x)
            x = F.linear(x, params[6], params[7])
            x = F.relu(x)
            x = F.linear(x, params[8], params[9])
            return x

    def get_parameters(self):
        # Return parameters in the order expected by forward()
        return [
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc3.weight,
            self.fc3.bias,
            self.fc4.weight,
            self.fc4.bias,
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
        # If params is None, use self's parameters; otherwise, use the provided ones.
        if params is None:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)
        else:
            # A simple functional implementation for our four-layer CNN.
            # We assume that params is a list in the order:
            # [conv1.weight, conv1.bias, bn1.weight, bn1.bias,
            #  conv2.weight, conv2.bias, bn2.weight, bn2.bias, ...,
            #  classifier.weight, classifier.bias]
            p = params  # shorthand
            # Layer 1
            out = F.conv2d(x, p[0], p[1], stride=1, padding=1)
            # We mimic BatchNorm using running stats from self.layer1[1] if needed.
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
            out = F.max_pool2d(out, kernel_size=2)

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
            out = F.max_pool2d(out, kernel_size=2)

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
            out = F.max_pool2d(out, kernel_size=2)

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
            out = F.max_pool2d(out, kernel_size=2)

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
        fast_weights = (
            self.net.get_parameters()
        )  # Keep using the original method for now

        # We can't transition to OrderedDict directly without changing more code
        # Make sure all parameters require gradients
        for i, w in enumerate(fast_weights):
            if not w.requires_grad:
                print(f"Warning: Parameter {i} does not require grad")
                w.requires_grad = True

        # Inner-loop adaptation
        for _ in range(self.args.update_step):
            logits = self.net.forward(x_spt, params=fast_weights)
            loss = F.cross_entropy(logits, y_spt)

            # Check if loss is valid
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
            if any(g is None for g in grads):
                print("Warning: Some gradients are None")
                # Create dummy gradients where needed
                grads = [
                    g if g is not None else torch.zeros_like(w)
                    for g, w in zip(grads, fast_weights)
                ]

            # Debug gradient norms

            # Apply gradient clipping
            grads = [torch.clamp(g, -0.5, 0.5) for g in grads]
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
        logits_q = self.net.forward(x_qry, params=fast_weights)
        qry_loss = F.cross_entropy(logits_q, y_qry)

        # Calculate accuracy
        pred = torch.argmax(logits_q, dim=1)
        correct = torch.eq(pred, y_qry).sum().item()
        acc = correct / float(len(y_qry))
        return qry_loss, acc

    # Then update the finetuning method to use the original parameter approach
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        Finetuning for evaluation. Returns accuracies after each update step.
        This procedure is done without computing higher-order gradients.
        """
        net_copy = copy.deepcopy(self.net)
        fast_weights = net_copy.get_parameters()  # Keep using original for now
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
            # Apply gradient clipping
            clipped_grads = [torch.clamp(g, -0.5, 0.5) for g in grads]

            # Update with clipped gradients
            fast_weights = [
                w - self.args.inner_lr * g for w, g in zip(fast_weights, clipped_grads)
            ]

            logits_q = net_copy.forward(x_qry, params=fast_weights)
            pred = torch.argmax(logits_q, dim=1)
            correct = torch.eq(pred, y_qry).sum().item()
            acc = correct / float(len(y_qry))
            accs.append(acc)

        return accs
