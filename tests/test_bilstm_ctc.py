import torch
from src.models.bilstm_ctc import BiLSTMCTC


def test_output_shape_and_logsoftmax():
    m = BiLSTMCTC(8, 10, hidden_size=16, num_layers=2).eval()
    lp, ol = m(torch.randn(3, 20, 8), torch.tensor([20, 15, 7]))
    assert lp.shape == (20, 3, 10) and ol.tolist() == [20, 15, 7]
    assert torch.allclose(lp.exp().sum(-1), torch.ones(20, 3), atol=1e-5)


def test_padding_is_ignored():
    m = BiLSTMCTC(8, 10, hidden_size=16, num_layers=2).eval()
    x = torch.randn(1, 12, 8)
    a, _ = m(x, torch.tensor([12]))
    padded = torch.cat([x, torch.randn(1, 30, 8) * 100], 1)
    b, _ = m(padded, torch.tensor([12]))
    assert torch.allclose(a[:12], b[:12], atol=1e-5)


def test_ctc_loss_trains():
    torch.manual_seed(0)
    m = BiLSTMCTC(8, 5, hidden_size=16, num_layers=1, dropout=0.0)
    ctc = torch.nn.CTCLoss(blank=4, zero_infinity=True)
    x = torch.randn(2, 15, 8); L = torch.tensor([15, 12])
    tg = torch.tensor([0, 1, 2, 3, 3]); tl = torch.tensor([3, 2])
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    first = None
    for _ in range(60):
        opt.zero_grad(); lp, ol = m(x, L); loss = ctc(lp, tg, ol, tl); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < first * 0.5
