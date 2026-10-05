import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile
import numpy as np
from plyfile import PlyData, PlyElement
import pytest
from agri3d.io.ply import PointCloudData, PLYReader
from agri3d.preprocessing.labels import decode_maizefield3d_labels
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.normalization import CloudNormalizer, restore_xyz
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.dataset import SegmentationDataset

COLORS=np.array([[0,0,0],[255,0,0],[0,1,0],[1,1,1],[0,0,1],[128,64,2]],dtype=np.uint8)
def cloud():
    return PointCloudData(np.arange(18,dtype=np.float32).reshape(6,3),{'colors':COLORS.copy(),'source_index':np.arange(6)})

def test_binary_rgb():
    c=cloud();out=decode_maizefield3d_labels(c)
    np.testing.assert_array_equal(out.attributes['label'],[1,0,0,0,0,0])
    assert out.attributes['label'].dtype==np.int64
    np.testing.assert_array_equal(out.attributes['colors'],COLORS)
    assert 'label' not in c.attributes
    out.attributes['colors'][0]=255
    np.testing.assert_array_equal(c.attributes['colors'],COLORS)

@pytest.mark.parametrize('colors',[None,np.zeros(6),np.zeros((6,4)),np.zeros((5,3)),np.zeros((6,3)),np.full((6,3),-1),np.full((6,3),256)])
def test_malformed_rgb(colors):
    c=cloud()
    if colors is None:del c.attributes['colors']
    else:c.attributes['colors']=colors
    with pytest.raises(ValueError,match='RGB annotations'):decode_maizefield3d_labels(c)

@pytest.mark.parametrize('method',['random','voxel','fps'])
def test_alignment(method):
    c=cloud();c.xyz[2]=np.nan
    cleaned=CloudFilter().remove_nan_inf(decode_maizefield3d_labels(c))
    normalized=CloudNormalizer(scale_to_unit_sphere=True).process(cleaned)
    out=FixedSizeSampler(9,method).sample(normalized,np.random.default_rng(2));idx=out.attributes['source_index']
    np.testing.assert_array_equal(out.attributes['colors'],COLORS[idx])
    np.testing.assert_array_equal(out.attributes['label'],np.all(COLORS[idx]==0,axis=1))
    np.testing.assert_allclose(restore_xyz(out),c.xyz[idx],atol=1e-5)
    assert out.attributes['valid_mask'].sum()==5

def write_ply(path,binary=False):
    d=np.zeros(60,dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    d['x']=np.arange(60)/100;d['z']=np.arange(60)/20
    for j,k in enumerate(('red','green','blue')):d[k]=np.tile(COLORS,(10,1))[:,j]
    PlyData([PlyElement.describe(d,'vertex')],text=not binary).write(str(path))

@pytest.mark.parametrize('binary',[False,True])
def test_rgb_dataset(tmp_path,binary):
    import torch
    from torch.utils.data import DataLoader
    path=tmp_path/'plant.ply';write_ply(path,binary)
    rs=[dict(path=path.name,sample_id='plant',group_id='g',split='train')]
    ds=SegmentationDataset(rs,tmp_path,'train',20,sampling='random',annotation_mode='maizefield3d')
    a=ds[0];batch=next(iter(DataLoader(ds,batch_size=1,num_workers=0)))
    assert a['points'].dtype==torch.float32 and a['labels'].dtype==torch.int64
    assert batch['points'].shape==(1,20,3)
    np.testing.assert_array_equal(a['labels'],a['source_index'].numpy()%6==0)
    ds.set_epoch(1);assert not torch.equal(a['source_index'],ds[0]['source_index'])
    for split in ('val','test'):
        val=SegmentationDataset([dict(rs[0],split=split)],tmp_path,split,80,annotation_mode='maizefield3d')
        before=val[0];val.set_epoch(3)
        assert torch.equal(before['source_index'],val[0]['source_index'])
        assert before['valid_mask'].sum()==60
    raw=PLYReader(path).load_point_cloud();assert 'label' not in raw.attributes
    np.testing.assert_array_equal(raw.attributes['colors'],np.tile(COLORS,(10,1)))
    with pytest.raises(ValueError,match='combined'):
        SegmentationDataset(rs,tmp_path,'train',annotation_mode='maizefield3d',label_map={0:0})

def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location('script_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('script,archive',[('download_ground_truth.py','datasets/FielGrwon_ZeaMays_SegmentedPCD_100k.zip'),('download_maize_data.py','datasets/FielGrwon_ZeaMays_RawPCD_100k.zip')])
@pytest.mark.parametrize('unsafe',[False,True])
def test_download_paths(tmp_path,monkeypatch,script,archive,unsafe):
    path=tmp_path/'a.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('../escape.ply' if unsafe else 'plants/a.ply','fixture')
    calls=[]
    def download(**kwargs):calls.append(kwargs);return str(path)
    monkeypatch.setitem(sys.modules,'huggingface_hub',types.SimpleNamespace(hf_hub_download=download))
    m=load_script(script)
    monkeypatch.setattr(sys,'argv',[script,'--revision','verified-commit','--output',str(tmp_path/'out')])
    if unsafe:
        with pytest.raises(ValueError,match='Unsafe archive'):m.main()
        assert not (tmp_path/'escape.ply').exists()
    else:m.main();assert (tmp_path/'out/plants/a.ply').exists()
    assert calls==[dict(repo_id='BGLab/MaizeField3D',repo_type='dataset',filename=archive,revision='verified-commit')]

def test_baseline_cli(tmp_path,monkeypatch):
    path=tmp_path/'plant.ply';write_ply(path)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps([dict(path=path.name,sample_id='plant',group_id='g',split='val')]))
    output=tmp_path/'metrics.json';m=load_script('evaluate_baseline.py')
    monkeypatch.setattr(sys,'argv',['evaluate_baseline.py','--manifest',str(manifest),'--root',str(tmp_path),'--points','80','--output',str(output)])
    m.main();r=json.loads(output.read_text())
    assert r['annotation_mode']=='maizefield3d' and r['metrics']['evaluated_points']==60
    assert 'fixed sampled points' in r['scope']
    for key in ('accuracy','mIoU','class_0_iou','class_1_iou','class_0_precision','class_1_recall','class_1_f1'):assert key in r['metrics']

@pytest.mark.parametrize('args',[['--annotation-mode','explicit'],['--encoding','unused.json']])
def test_cli_rejects_ambiguous_encoding(tmp_path,monkeypatch,args):
    m=load_script('evaluate_baseline.py')
    monkeypatch.setattr(sys,'argv',['evaluate_baseline.py','--manifest','m.json','--root',str(tmp_path),'--output','out.json',*args])
    with pytest.raises(SystemExit) as exc:m.main()
    assert exc.value.code==2
