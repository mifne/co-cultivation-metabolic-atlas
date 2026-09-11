import pytest


def test_device_condition_skips_bodies_and_resets_between_replays():
    cp=pytest.importorskip('cupy')
    from src.gpu_conditional_capture import ConditionalCapture
    conditional=ConditionalCapture()
    stopped=cp.zeros(1,dtype=bool);failed=stopped.copy();count=cp.zeros(1)
    pool=cp.cuda.MemoryPool();stream=cp.cuda.Stream(non_blocking=True)
    # Compile the elementary operations before stream capture.
    count+=1.;stopped[:]=count>=3;count[:]=0;stopped[:]=False
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.using_allocator(pool.malloc),stream:
        temp=count>=3;stream.synchronize();del temp
        stream.begin_capture()
        count[:]=0;stopped[:]=False
        for _ in range(8):
            with conditional.iteration(stopped,failed):
                count+=1.;stopped[:]=count>=3
        graph=stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    assert count.get()[0]==3
    graph.launch(stream);stream.synchronize()
    assert count.get()[0]==3
