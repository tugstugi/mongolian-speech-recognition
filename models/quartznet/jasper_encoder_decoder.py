#
# original code: https://github.com/NVIDIA/NeMo/blob/r0.8.2/collections/nemo_asr/nemo_asr/jasper.py
#


import torch.nn as nn
from .jasper_block import JasperBlock, jasper_activations, init_weights


class JasperEncoderDecoder(nn.Module):
    """
    Jasper Encoder creates the pre-processing (prologue), Jasper convolution
    block, and the first 3 post-processing (epilogue) layers as described in
    Jasper (https://arxiv.org/abs/1904.03288)

    Args:
        jasper (list): A list of dictionaries. Each element in the list
            represents the configuration of one Jasper Block. Each element
            should contain::

                {
                    # Required parameters
                    'filters' (int) # Number of output channels,
                    'repeat' (int) # Number of sub-blocks,
                    'kernel' (int) # Size of conv kernel,
                    'stride' (int) # Conv stride
                    'dilation' (int) # Conv dilation
                    'dropout' (float) # Dropout probability
                    'residual' (bool) # Whether to use residual or not.
                    # Optional parameters
                    'residual_dense' (bool) # Whether to use Dense Residuals
                        # or not. 'residual' must be True for 'residual_dense'
                        # to be enabled.
                        # Defaults to False.
                    'separable' (bool) # Whether to use separable convolutions.
                        # Defaults to False
                    'groups' (int) # Number of groups in each conv layer.
                        # Defaults to 1
                    'heads' (int) # Sharing of separable filters
                        # Defaults to -1
                    'tied' (bool)  # Whether to use the same weights for all
                        # sub-blocks.
                        # Defaults to False
                }

        activation (str): Activation function used for each sub-blocks. Can be
            one of ["hardtanh", "relu", "selu"].
        feat_in (int): Number of channels being input to this module
        num_classes (int): number of vocab including the blank character
        normalization_mode (str): Normalization to be used in each sub-block.
            Can be one of ["batch", "layer", "instance", "group"]
            Defaults to "batch".
        residual_mode (str): Type of residual connection.
            Can be "add" or "max".
            Defaults to "add".
        norm_groups (int): Number of groups for "group" normalization type.
            If set to -1, number of channels is used.
            Defaults to -1.
        conv_mask (bool): Controls the use of sequence length masking prior
            to convolutions.
            Defaults to True.
        frame_splicing (int): Defaults to 1.
        init_mode (str): Describes how neural network parameters are
            initialized. Options are ['xavier_uniform', 'xavier_normal',
            'kaiming_uniform','kaiming_normal'].
            Defaults to "xavier_uniform".
    """

    def __init__(
            self,
            jasper,
            activation,
            feat_in,
            num_classes,
            normalization_mode="batch",
            residual_mode="add",
            norm_groups=-1,
            conv_mask=True,
            frame_splicing=1,
            init_mode='xavier_uniform',
            **kwargs
    ):
        super(JasperEncoderDecoder, self).__init__()

        activation = jasper_activations[activation]()
        feat_in = feat_in * frame_splicing

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            tied = lcfg.get('tied', False)
            heads = lcfg.get('heads', -1)
            encoder_layers.append(
                JasperBlock(feat_in,
                            lcfg['filters'],
                            repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'],
                            stride=lcfg['stride'],
                            dilation=lcfg['dilation'],
                            dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0,
                            residual=lcfg['residual'],
                            groups=groups,
                            separable=separable,
                            heads=heads,
                            residual_mode=residual_mode,
                            normalization=normalization_mode,
                            norm_groups=norm_groups,
                            tied=tied,
                            activation=activation,
                            residual_panes=dense_res,
                            conv_mask=conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(1024, num_classes,
                      kernel_size=1, bias=True)
            )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length):
        s_input, length = self.encoder(([audio_signal], length))
        # BxCxT
        return self.decoder_layers(s_input[-1]), length

    def load_nvidia_nemo_weights(self, encoder_weight_path, decoder_weight_path, map_location='cpu'):
        import torch
        encoder_weight = torch.load(encoder_weight_path, map_location=map_location)
        new_encoder_weight = {}
        for k, v in encoder_weight.items():
            k = k.replace('mconv', 'conv')
            if len(v.shape) == 3:
                k = k.replace('.conv.weight', '.weight')
            new_encoder_weight[k] = v
        if decoder_weight_path:
            decoder_weight = torch.load(decoder_weight_path, map_location=map_location)
            encoder_weight.update(decoder_weight)
        self.load_state_dict(new_encoder_weight, strict=False)
