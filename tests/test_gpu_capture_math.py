import numpy as np
import pytest


def test_fp64_primitives_and_capture():
    cp=pytest.importorskip('cupy')
    from cupyx.scipy.sparse import csr_matrix
    from src.gpu_capture_math import CaptureMath
    math=CaptureMath();rng=np.random.default_rng(87)
    a=cp.asarray(rng.normal(size=(3,17,11)));b=cp.asarray(rng.normal(size=(3,11,5)))
    np.testing.assert_allclose(math.mm(a,b).get(),(a@b).get(),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(math.mm(a[0],b).get(),(a[0]@b).get(),rtol=1e-12,atol=1e-12)
    sparse=csr_matrix(a[0]);x=b[0]
    np.testing.assert_allclose(math.mm(sparse,x).get(),(sparse@x).get(),rtol=1e-12,atol=1e-12)
    a=cp.asarray(rng.normal(size=(3,12,12)))+cp.eye(12)[None]*12
    b=cp.asarray(rng.normal(size=(3,12,2)))
    np.testing.assert_allclose(math.solve(a,b).get(),cp.linalg.solve(a,b).get(),atol=1e-12)
    # Capture only primitives with fixed persistent buffers. The caller must
    # retain its private allocation pool for the lifetime of the graph.
    pool=cp.cuda.MemoryPool()
    stream=cp.cuda.Stream(non_blocking=True)
    with cp.cuda.using_allocator(pool.malloc),stream:
        warm=math.solve(a,b);product=math.mm(a,warm)
        stream.synchronize()
        del warm,product
        stream.begin_capture()
        solution=math.solve(a,b);product=math.mm(a,solution)
        graph=stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    np.testing.assert_allclose(product.get(),b.get(),atol=1e-12)
    b*=2
    cp.cuda.get_current_stream().synchronize()
    graph.launch(stream);stream.synchronize()
    np.testing.assert_allclose(product.get(),b.get(),atol=1e-12)
def test_identity_padding_active_prefix_retains_all_rhs():
    import pytest
    cp=pytest.importorskip('cupy')
    import numpy as np
    from src.gpu_capture_math import CaptureMath
    math=CaptureMath()
    a=cp.broadcast_to(cp.eye(8),(2,8,8)).copy()
    a[:,:2,:2]=cp.array([[3.,1.],[1.,2.]])
    b=cp.arange(32,dtype=cp.float64).reshape(2,8,2)
    result=math.solve(a,b,active_size=cp.array([2],dtype=cp.int32))
    np.testing.assert_allclose(result.get(),cp.linalg.solve(a,b).get(),atol=1e-12)


@pytest.mark.parametrize('size,active',[(7,7),(128,77),(384,129),(8,0)])
def test_reused_lu_forward_transpose_and_identity_padding(size,active):
    cp=pytest.importorskip('cupy')
    from src.gpu_capture_math import CaptureMath
    math=CaptureMath();rng=np.random.default_rng(120+size)
    a=np.broadcast_to(np.eye(size),(2,size,size)).copy()
    if active:
        block=rng.normal(size=(2,active,active))+active*np.eye(active)
        # Row permutation forces pivoting and exercises transpose permutations.
        a[:,:active,:active]=block[:,rng.permutation(active)]
    b=rng.normal(size=(2,size,3));da=cp.asarray(a);db=cp.asarray(b)
    factor=math.factor(da,cp.array([active],dtype=cp.int32))
    for transpose in (False,True):
        out=math.solve_factored(factor,db,transpose=transpose).get()
        expected=np.linalg.solve(a.transpose(0,2,1) if transpose else a,b)
        np.testing.assert_allclose(out,expected,rtol=1e-11,atol=1e-11)


def test_reused_lu_capture_refreshes_changed_matrix_and_rhs():
    cp=pytest.importorskip('cupy')
    from src.gpu_capture_math import CaptureMath
    math=CaptureMath();rng=np.random.default_rng(15)
    a=cp.asarray(rng.normal(size=(2,12,12))+12*np.eye(12));b=cp.asarray(rng.normal(size=(2,12,2)))
    pool=cp.cuda.MemoryPool();stream=cp.cuda.Stream(non_blocking=True)
    with cp.cuda.using_allocator(pool.malloc),stream:
        factor=math.factor(a);out=math.solve_factored(factor,b,True)
        stream.synchronize();del factor,out
        stream.begin_capture();factor=math.factor(a);out=math.solve_factored(factor,b,True)
        graph=stream.end_capture();graph.launch(stream)
    stream.synchronize()
    np.testing.assert_allclose(out.get(),np.linalg.solve(a.get().transpose(0,2,1),b.get()),atol=1e-11)
    a+=cp.eye(12)[None];b*=3;cp.cuda.get_current_stream().synchronize()
    graph.launch(stream);stream.synchronize()
    np.testing.assert_allclose(out.get(),np.linalg.solve(a.get().transpose(0,2,1),b.get()),atol=1e-11)


def test_private_cublas_handle_can_be_released_idempotently():
    pytest.importorskip('cupy')
    from src.gpu_capture_math import CaptureMath
    math=CaptureMath();assert math.handle.value
    math.close();assert math.handle.value is None
    math.close()
