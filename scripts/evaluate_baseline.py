import argparse
import json
from pathlib import Path
from time import perf_counter
from agri3d.training.dataset import SegmentationDataset
from agri3d.training.evaluate import SegmentationEvaluator
from agri3d.io.ply import PointCloudData
from agri3d.geometry.clustering import GeometricBaseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--encoding', required=True, help='JSON: rgb_to_class or label_to_class')
    parser.add_argument('--split', choices=['train','val','test'], default='val')
    parser.add_argument('--points', type=int, default=10000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    encoding = json.loads(Path(args.encoding).read_text())
    colors = {tuple(map(int,k.split(','))):v for k,v in encoding['rgb_to_class'].items()} if 'rgb_to_class' in encoding else None
    labels = {int(k):v for k,v in encoding['label_to_class'].items()} if 'label_to_class' in encoding else None
    ds = SegmentationDataset(args.manifest, args.root, args.split, args.points,
                             color_map=colors, label_map=labels)
    evaluator = SegmentationEvaluator()
    records = []
    for item in ds:
        # Baseline radius/DBSCAN eps are meters; restore scale before geometric inference.
        valid = item['valid_mask'].numpy()
        xyz = item['points'].numpy()[valid] * item['scale'].item()
        cloud = PointCloudData(xyz.astype('float32'))
        start = perf_counter()
        predictions = GeometricBaseline().segment(cloud)
        elapsed = (perf_counter()-start)*1000
        evaluator.update(predictions, item['labels'].numpy()[valid])
        records.append({'sample_id':item['sample_id'], 'baseline_inference_ms':elapsed})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'split':args.split, 'scope':'fixed sampled points; not full-cloud coverage',
                                 'metrics':evaluator.compute(), 'samples':records}, indent=2))

if __name__ == '__main__':
    main()
