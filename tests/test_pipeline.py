import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from agri3d.io.ply import PointCloudData, PLYReader
from agri3d.preprocessing.labels import decode_labels
from agri3d.preprocessing.filtering import CloudFilter
from agri3d.preprocessing.normalization import CloudNormalizer, restore_xyz
from agri3d.preprocessing.sampling import FixedSizeSampler
from agri3d.training.splits import make_group_split, validate_manifest
from agri3d.training.evaluate import SegmentationEvaluator

def cloud(n=12):
    return PointCloudData(np.arange(n*3,dtype=np.float32).reshape(n,3),
        {'label':np.arange(n)%2, 'source_index':np.arange(n)})

@pytest.mark.parametrize('method',['random','voxel','fps'])
@pytest.mark.parametrize('n,k',[(12,5),(5,5),(3,8)])
def test_sampling_alignment(method,n,k):
    c=cloud(n); out=FixedSizeSampler(k,method,2).sample(c,np.random.default_rng(4))
    idx=out.attributes['source_index']
    assert out.xyz.shape==(k,3)
    np.testing.assert_array_equal(out.xyz,c.xyz[idx])
    np.testing.assert_array_equal(out.attributes['label'],c.attributes['label'][idx])
    assert out.attributes['valid_mask'].sum()==min(n,k)
    assert len(set(idx[out.attributes['valid_mask']]))==min(n,k)
    other=FixedSizeSampler(k,method,2).sample(c,np.random.default_rng(4))
    np.testing.assert_array_equal(idx,other.attributes['source_index'])

def test_fps_identical_points():
    c=cloud();c.xyz[:]=0
    s=FixedSizeSampler(5,'fps').sample(c,np.random.default_rng(0))
    assert len(set(s.attributes['source_index']))==5

def test_filter_and_voxel_preserve_labels():
    c=cloud();c.xyz[2]=np.nan
    f=CloudFilter(voxel_size=10)
    cleaned=f.remove_nan_inf(c)
    assert len(c.xyz)==12 and len(cleaned.xyz)==11
    out=f.voxel_downsample(cleaned)
    np.testing.assert_array_equal(out.attributes['label'],c.attributes['label'][out.attributes['source_index']])
    c.xyz[:]=np.nan
    with pytest.raises(ValueError): f.remove_nan_inf(c)

def test_normalization_roundtrip():
    c=cloud();out=CloudNormalizer(scale_to_unit_sphere=True).process(c)
    np.testing.assert_allclose(restore_xyz(out),c.xyz,atol=1e-5)
    with pytest.raises(ValueError): CloudNormalizer().process(out)
    c.xyz[:]=7
    assert np.isfinite(CloudNormalizer(scale_to_unit_sphere=True).process(c).xyz).all()

def test_explicit_decoding():
    c=cloud(2);c.attributes['colors']=np.array([[255,0,0],[0,255,0]])
    with pytest.raises(ValueError): decode_labels(c)
    with pytest.raises(ValueError): decode_labels(c,{(255,0,0):1})
    np.testing.assert_array_equal(decode_labels(c,{(255,0,0):1,(0,255,0):0}).attributes['label'],[1,0])
    np.testing.assert_array_equal(decode_labels(c,label_map={0:1,1:0}).attributes['label'],[1,0])

def records():
    return [dict(path=f'{i}.ply',sample_id=str(i),group_id=str(i//2)) for i in range(20)]

def test_group_splits():
    result=make_group_split(records())
    assert result==make_group_split(list(reversed(records())))
    assert {r['split'] for r in result}=={'train','val','test'}
    result[1]['split']='test' if result[0]['split']!='test' else 'val'
    with pytest.raises(ValueError):validate_manifest(result)

def test_global_metrics_and_mask():
    e=SegmentationEvaluator()
    e.update(np.array([0,1,1]),np.array([0,0,1]))
    e.update(np.array([0,1]),np.array([1,0]),np.array([True,False]))
    m=e.compute()
    assert m['accuracy']==0.5
    assert m['mIoU']==pytest.approx(1/3)
    assert m['class_1_precision']==0.5
    a=e.calculate_metrics(np.array([0]),np.array([0]))
    assert a['class_1_iou'] is None and a['mIoU']==1
    with pytest.raises(ValueError):e.update(np.array([0]),np.array([[0]]))
    with pytest.raises(ValueError):e.update(np.array([2]),np.array([0]))

def write_ply(path,n=20,binary=False):
    data=np.zeros(n,dtype=[('x','f4'),('y','f4'),('z','f4'),('label','i4'),('red','u1'),('green','u1'),('blue','u1')])
    data['x']=np.arange(n);data['z']=np.arange(n)/2;data['label']=np.arange(n)%2
    PlyData([PlyElement.describe(data,'vertex')],text=not binary).write(str(path))

@pytest.mark.parametrize('binary',[False,True])
def test_reader(tmp_path,binary):
    f=tmp_path/'a.ply';write_ply(f,binary=binary)
    c=PLYReader(f).load_point_cloud()
    assert c.xyz.dtype==np.float32
    assert 'instance_id' not in c.attributes
    assert len(c.attributes['label'])==20

def test_dataset_dataloader(tmp_path):
    import torch
    from torch.utils.data import DataLoader
    from agri3d.training.dataset import SegmentationDataset
    rs=[]
    for i,split in enumerate(['train','train','val','test']):
        write_ply(tmp_path/f'{i}.ply')
        rs.append(dict(path=f'{i}.ply',sample_id=str(i),group_id=str(i),split=split))
    ds=SegmentationDataset(rs,tmp_path,'train',num_points=8,label_map={0:0,1:1},sampling='random')
    batch=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
    assert batch['points'].shape==(2,8,3) and batch['labels'].dtype==torch.int64
    np.testing.assert_array_equal(ds[0]['labels'],ds[0]['source_index']%2)
    before=ds[0]['source_index'].clone();ds.set_epoch(1)
    assert not torch.equal(before,ds[0]['source_index'])
    val=SegmentationDataset(rs,tmp_path,'val',8,label_map={0:0,1:1},sampling='random')
    before=val[0]['source_index'].clone();val.set_epoch(9)
    assert torch.equal(before,val[0]['source_index'])
    padded=SegmentationDataset(rs,tmp_path,'test',30,label_map={0:0,1:1})[0]
    assert padded['valid_mask'].sum()==20
    with pytest.raises(ValueError):SegmentationDataset(rs,tmp_path,'train',8)[0]

def test_sor_and_baseline():
    from agri3d.geometry.clustering import GeometricBaseline
    rng=np.random.default_rng(2)
    c=PointCloudData(rng.normal(0,.01,(100,3)).astype(np.float32),{'source_index':np.arange(100)})
    out=CloudFilter().process(c)
    np.testing.assert_array_equal(out.xyz,c.xyz[out.attributes['source_index']])
    pred=GeometricBaseline().segment(out)
    assert pred.shape==(len(out.xyz),) and np.isin(pred,[0,1]).all()

@pytest.mark.parametrize('k',[10000,50000,100000])
def test_contract_point_counts(k):
    out=FixedSizeSampler(k).sample(cloud(10),np.random.default_rng(0))
    assert out.xyz.shape==(k,3) and out.attributes['valid_mask'].sum()==10


def test_multiprocess_loader(tmp_path):
    import socket
    try:
        sock = socket.socket(socket.AF_UNIX); sock.close()
    except PermissionError:
        pytest.skip("Workspace blocks Unix sockets required for tensor transfer")
    from torch.utils.data import DataLoader
    from agri3d.training.dataset import SegmentationDataset
    write_ply(tmp_path/'a.ply')
    rs=[dict(path='a.ply',sample_id='a',group_id='a',split='train')]
    ds=SegmentationDataset(rs,tmp_path,'train',8,label_map={0:0,1:1})
    batch=next(iter(DataLoader(ds,batch_size=1,num_workers=2,multiprocessing_context='spawn',timeout=30)))
    assert batch['points'].shape==(1,8,3)
