"""Read a CAD STL without changing it; distinguish sheet gauge from bent depth."""
import struct
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from door8.core import REPO, dump, sha256


def measure(path):
    data=path.read_bytes();n=struct.unpack('<I',data[80:84])[0]
    if len(data)!=84+50*n:raise ValueError('This reader requires a binary STL with a complete facet table')
    dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attribute','<u2')])
    facets=np.frombuffer(data,offset=84,count=n,dtype=dtype);v=facets['v'].astype(float)
    cross=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);area=np.linalg.norm(cross,axis=1)/2
    normals=cross/(2*area[:,None]+1e-30)
    axis=normals[int(np.argmax(area))].copy()
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    parallel=np.abs(normals@axis)>1-1e-7;planes=defaultdict(float)
    for distance,ar in zip(v.mean(1)[parallel]@axis,area[parallel]):planes[round(float(distance),4)]+=float(ar)
    ranked=sorted(planes.items(),key=lambda x:-x[1]);a,b=ranked[:2]
    thickness=abs(a[0]-b[0]);xyz=v.reshape(-1,3)
    return {'file':str(path),'sha256':sha256(path),'facets':n,'extents_stl_units':np.ptp(xyz,axis=0).tolist(),
        'reference_plane_unit_normal':axis.tolist(),'two_largest_parallel_planes':[{'offset':o,'area':ar} for o,ar in [a,b]],
        'separation_stl_units':thickness,'normal_parallel_tolerance':1e-7,
        'interpretation':'Nominal main-sheet thickness from the two largest parallel plane groups. Not total bent-part depth or a measurement of the manufactured part.'}


if __name__=='__main__':
    p=REPO/'cad/door_stl/E25_door_LH_FRT.stl';r=measure(p)
    r.update(expected_cad_width_height_mm=[828,1140],
        unit_basis='STL has no explicit unit. X/Z extents agree with the known 828 x 1140 mm CAD dimensions; interpret STL units as mm.',
        nominal_main_sheet_thickness_mm=r['separation_stl_units'],
        scope='One representative STL checked directly. User states all target door sheets have the same thickness.')
    dump('reports/stl_thickness.json',r);print(r)
