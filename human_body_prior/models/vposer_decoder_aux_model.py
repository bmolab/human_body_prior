from torch import nn

from human_body_prior.models.vposer_model import ContinuousRotReprDecoder, VPoser
from human_body_prior.tools.rotation_tools import matrot2aa


class VPoserDecoderTorqueProxy(VPoser):
    """VPoser with a decoder-side auxiliary torque-proxy prediction head."""

    def __init__(self, model_ps):
        super(VPoserDecoderTorqueProxy, self).__init__(model_ps)

        num_neurons = model_ps.model_params.num_neurons

        self.decoder_net = None
        # Shared decoder features feed both pose reconstruction and torque prediction.
        self.decoder_shared = nn.Sequential(
            nn.Linear(self.latentD, num_neurons),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_neurons, num_neurons),
            nn.LeakyReLU(),
        )
        # Pose head keeps the same 6D rotation reconstruction target as VPoser.
        self.pose_decoder_head = nn.Linear(num_neurons, self.num_joints * 6)
        self.continuous_rot_repr_decoder = ContinuousRotReprDecoder()

        torque_head_ps = model_ps.model_params.toDict().get("torque_proxy_head", {})
        hiddenD = int(torque_head_ps.get("hiddenD", max(16, self.latentD)))
        dropout = float(torque_head_ps.get("dropout", 0.1))
        # Auxiliary head predicts one torque_proxy scalar from decoder features.
        self.torque_proxy_head = nn.Sequential(
            nn.Linear(num_neurons, hiddenD),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenD, 1),
        )

    def decode(self, Zin):
        bs = Zin.shape[0]

        decoder_features = self.decoder_shared(Zin)
        pose_6d = self.pose_decoder_head(decoder_features)
        prec = self.continuous_rot_repr_decoder(pose_6d)

        # Keep output keys compatible with the existing trainer torque loss.
        return {
            "pose_body": matrot2aa(prec.view(-1, 3, 3)).view(bs, -1, 3),
            "pose_body_matrot": prec.view(bs, -1, 9),
            "pred_torque_proxy": self.torque_proxy_head(decoder_features).view(-1),
        }
