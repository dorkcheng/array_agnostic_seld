# -*- coding: utf-8 -*-
"""诊断：用官方 PSELDNets mACCDOA-CNN14-Conformer 权重（含原 170 类头）在仿真理想 FOA 上推理，
比较不同 IV 通道顺序下 DOA 误差，并核对 mel 滤波器与官方一致。
用法：python scripts/check_pseldnets_order.py CKPT REPO H5 [H5 ...]"""
import os, sys, torch, numpy as np, h5py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT, REPO, H5S = sys.argv[1], sys.argv[2], sys.argv[3:]
from models.backbone import PSELDNetsBackbone
from seld.features import LogmelIV, stft_reim, mel_filterbank
from seld.utils import angular_distance_deg, sph2cart
ck = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
fb = ck["af_extractor.mel_scale.fb"].numpy(); print("mel fb diff vs ours:", np.abs(fb - mel_filterbank(24000,1024,64)).max(), fb.shape)
print("window", ck["af_extractor.stft_extractor.window"][:3])
bb = PSELDNetsBackbone(5, REPO, CKPT).eval()
fc = torch.nn.Linear(2048, 9*170); fc.weight.data = ck["net._orig_mod.fc.weight"]; fc.bias.data = ck["net._orig_mod.fc.bias"]
for order in [(3,1,2),(1,2,3)]:
    ext = LogmelIV(iv_xyz_index=order); errs=[]
    for path in H5S:
      with h5py.File(path) as h5:
        for sid in sorted(h5.keys()):
            g=h5[sid]; rows=g["labels"][()]
            foa=torch.from_numpy(g["foa"][()].astype(np.float32))[None]*20
            re,im=stft_reim(foa); f=ext(re,im)[:,:,:300]
            with torch.no_grad():
                x=bb.convs(bb.scalar(f)).mean(3).transpose(1,2); x=bb.decoder(x)
                x=torch.nn.functional.interpolate(x.transpose(1,2),size=30).transpose(1,2)
                o=torch.tanh(fc(x)).view(30,3,3,170).numpy()
            for r in rows:
                fr=int(r[0])
                if (rows[:,0]==fr).sum()!=1: continue
                v=o[fr]; n=np.linalg.norm(v,axis=1)  # (3,170)
                k,c=np.unravel_index(n.argmax(), n.shape)
                errs.append(angular_distance_deg(v[k,:,c], sph2cart(r[3],r[4])))
    print(order, "frames", len(errs), "median DOA err %.1f deg" % np.median(errs), "mean %.1f" % np.mean(errs))
