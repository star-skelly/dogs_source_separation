import cv2
from starlet_att_markov import StarletAttentionAggregatorWrapper
from m2n2.src.m2n2_model import M2N2SegmentationModel
import sys
sys.path.append('../')
from double_gmm import subset

attn_agg = StarletAttentionAggregatorWrapper(subset_df=subset, sigma_feat=1.1, window_radius=2)
points = [(300, 175), (135, 140), (200, 150), (200, 286)]
points_in_segment = [True, True, True, False]

# Predict
img_rgb_for_display = cv2.imread('../m2n2/images/image.jpg')[:, :, ::-1]
model = M2N2SegmentationModel(attn_agg)
attn = attn_agg.extract_attention(img_rgb_for_display)
print("attn.shape:", attn.shape)
bh, bw, h, w = attn.shape
A = attn.reshape(bh * bw, h * w)
print("A.shape:", A.shape)
print("row sums:", A.sum(dim=1).min().item(), A.sum(dim=1).max().item())
seg = model.segment(img_rgb_for_display, points, points_in_segment)