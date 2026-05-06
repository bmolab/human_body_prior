from torch import nn

from human_body_prior.models.vposer_model import VPoser


class VPoserTorqueProxy(VPoser):
    """VPoser with an auxiliary latent-to-torque-proxy prediction head."""

    def __init__(self, model_ps):
        super(VPoserTorqueProxy, self).__init__(model_ps)

        torque_head_ps = model_ps.model_params.toDict().get('torque_proxy_head', {})
        hiddenD = int(torque_head_ps.get('hiddenD', max(16, self.latentD)))
        dropout = float(torque_head_ps.get('dropout', 0.1))
        self.torque_proxy_head = nn.Sequential(
            nn.Linear(self.latentD, hiddenD),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiddenD, 1),
        )

    def forward(self, pose_body):
        decode_results = super(VPoserTorqueProxy, self).forward(pose_body)
        decode_results['pred_torque_proxy'] = self.torque_proxy_head(decode_results['poZ_body_mean']).view(-1)
        return decode_results
