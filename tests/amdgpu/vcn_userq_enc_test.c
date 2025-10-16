
// SPDX-License-Identifier: MIT
/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "CUnit/Basic.h"

#include "amdgpu_drm.h"
#include "amdgpu_internal.h"
#include "amdgpu_test.h"
#include "util_math.h"
#include "xf86drm.h"

#include "frame.h"
#include "vcn_tests.h"


#define upper_32_bits(n) ((__u32)(((n) >> 16) >> 16))
#define lower_32_bits(n) ((__u32)(n))

#define PAGE_SIZE				4096
#define USERMODE_QUEUE_SIZE			(PAGE_SIZE * 32)
#define ENCODE_BUF_SIZE			(PAGE_SIZE * 32)

#define ALIGNMENT				4096
#define VCN_DOORBELL_INDEX			0
#define AMDGPU_USERQ_BO_WRITE			1

#define VCN_ENC_CMD_NO_OP			0x00000000
#define VCN_ENC_CMD_END			0x00000001
#define VCN_ENC_CMD_IB				0x00000002
#define VCN_ENC_CMD_FENCE			0x00000003
#define VCN_PRTED_FENCE_SIG_CMD		0x00000010
#define VCN_PRTED_FENCE_WAIT_CMD		0x00000011
#define VCN_HDP_FLUSH_CMD			0x00000012

#define DECODE_CMD_MSG_BUFFER					0x00000000
#define DECODE_CMD_DPB_BUFFER					0x00000001
#define DECODE_CMD_DECODING_TARGET_BUFFER			0x00000002
#define DECODE_CMD_FEEDBACK_BUFFER				0x00000003
#define DECODE_CMD_PROB_TBL_BUFFER				0x00000004
#define DECODE_CMD_SESSION_CONTEXT_BUFFER			0x00000005
#define DECODE_CMD_BITSTREAM_BUFFER				0x00000100
#define DECODE_CMD_IT_SCALING_TABLE_BUFFER			0x00000204
#define DECODE_CMD_CONTEXT_BUFFER				0x00000206

#define DECODE_IB_PARAM_DECODE_BUFFER				(0x00000001)

#define DECODE_CMDBUF_FLAGS_MSG_BUFFER				(0x00000001)
#define DECODE_CMDBUF_FLAGS_DPB_BUFFER				(0x00000002)
#define DECODE_CMDBUF_FLAGS_BITSTREAM_BUFFER			(0x00000004)
#define DECODE_CMDBUF_FLAGS_DECODING_TARGET_BUFFER		(0x00000008)
#define DECODE_CMDBUF_FLAGS_FEEDBACK_BUFFER			(0x00000010)
#define DECODE_CMDBUF_FLAGS_IT_SCALING_BUFFER			(0x00000200)
#define DECODE_CMDBUF_FLAGS_CONTEXT_BUFFER			(0x00000800)
#define DECODE_CMDBUF_FLAGS_PROB_TBL_BUFFER			(0x00001000)
#define DECODE_CMDBUF_FLAGS_SESSION_CONTEXT_BUFFER		(0x00100000)

static amdgpu_device_handle device_handle;
static struct  amdgpu_vcn_bo doorbell, rb_base, wptr, rptr, ib_ptr, agdb_doorbell;
static struct  amdgpu_vcn_bo fence;
static uint32_t major_version;
static uint32_t minor_version;
static uint32_t q_id, db_handle, agdb_handle;
static struct drm_amdgpu_userq_mqd_vcn mqd;
static uint32_t *ib_checksum;
static uint32_t *ib_size_in_dw;
static uint32_t *ib_cpu, *wptr_cpu;
static uint32_t *wptr_va_cpu, *rptr_va_cpu;
static uint32_t *doorbell_ptr, *agdb_doorbell_ptr;
static uint64_t  *rptr_va_cpu_64;

#if FENCE_LEGACY
static uint32_t *fence_cpu;
#endif

static struct amdgpu_vcn_bo enc_buf;
static struct amdgpu_vcn_bo cpb_buf;
static uint32_t rb_len;

static uint32_t gWidth, gHeight, gSliceType;
static uint32_t enc_task_id;
static bool vcn_unified_ring = false;
static uint32_t vcn_ip_version_major;
static uint32_t vcn_ip_version_minor;

static uint32_t timeline_syncobj_handle;
static uint64_t point;

static void amdgpu_vcn_userqueue_create(void);
static void amdgpu_vcn_userqueue_destroy(void);
static void amdgpu_cs_vcn_enc_create(void);
static void amdgpu_cs_vcn_enc_encode(void);
static void amdgpu_cs_vcn_enc_destroy(void);

static int verify_checksum(uint8_t *buffer, uint32_t buffer_size);

int verify_checksum(uint8_t *buffer, uint32_t buffer_size)
{
	uint32_t buffer_pos = 0;
	int done = 0;
	h264_decode dec;

	memset(&dec, 0, sizeof(h264_decode));
	if (buffer_size == 0)
		return -1;

	do {
		uint32_t ret;

		ret = h264_find_next_start_code(buffer + buffer_pos,
				 buffer_size - buffer_pos);
		if (ret == 0) {
			done = 1;
			if (buffer_pos == 0) {
				fprintf(stderr,
				 "couldn't find start code in buffer from 0\n");
			}
		} else {
		/* have a complete NAL from buffer_pos to end */
			if (ret > 3) {
				uint32_t nal_len;
				bufferInfo bufinfo;

				nal_len = remove_03(buffer + buffer_pos, ret);
				bufinfo.decBuffer = buffer + buffer_pos +
						(buffer[buffer_pos + 2] == 1 ? 3 : 4);
				bufinfo.decBufferSize = (nal_len -
						(buffer[buffer_pos + 2] == 1 ? 3 : 4)) *
						8;
				bufinfo.end = buffer + buffer_pos + nal_len;
				bufinfo.numOfBitsInBuffer = 8;
				bufinfo.decData = *bufinfo.decBuffer;
				h264_parse_nal(&dec, &bufinfo);
			}
			buffer_pos += ret;	/*  buffer_pos points to next code */
		}
	} while (done == 0);

	if ((dec.pic_width == gWidth) &&
		(dec.pic_height == gHeight) &&
		(dec.slice_type == gSliceType))
		return 0;
	else
		return -1;
}

static void check_result(struct amdgpu_vcn_bo fb_buf, struct amdgpu_vcn_bo bs_buf, int frame_type)
{
	uint32_t *fb_ptr;
	uint8_t *bs_ptr;
	uint32_t size;
	int r;

	r = amdgpu_bo_cpu_map(fb_buf.handle, (void **)&fb_buf.ptr);
	CU_ASSERT_EQUAL(r, 0);
	fb_ptr = (uint32_t *)fb_buf.ptr;
	size = fb_ptr[6];

	r = amdgpu_bo_cpu_unmap(fb_buf.handle);
	CU_ASSERT_EQUAL(r, 0);
	r = amdgpu_bo_cpu_map(bs_buf.handle, (void **)&bs_buf.ptr);
	CU_ASSERT_EQUAL(r, 0);

	bs_ptr = (uint8_t *)bs_buf.ptr;
	r = verify_checksum(bs_ptr, size);
	CU_ASSERT_EQUAL(r, 0);
	r = amdgpu_bo_cpu_unmap(bs_buf.handle);

}

static void amdgpu_cs_vcn_ib_zero_count(int *len, int num)
{
	for (int i = 0; i < num; i++)
		ib_cpu[(*len)++] = 0;
}

CU_TestInfo userq_vcn_enc_tests[] = {
	{"VCN ENC create", amdgpu_cs_vcn_enc_create},
	{"VCN ENC encode", amdgpu_cs_vcn_enc_encode},
	{"VCN ENC destroy", amdgpu_cs_vcn_enc_destroy},
	CU_TEST_INFO_NULL,
};

CU_BOOL suite_vcn_userq_enc_tests_enable(void)
{
	struct drm_amdgpu_info_hw_ip info;
	int r;

	r = amdgpu_device_initialize(drm_amdgpu[0], &major_version,
				     &minor_version, &device_handle);
	if (r) {
		if ((r == -EACCES) && (errno == EACCES))
			printf("\n\nError:%s. "
			       "Hint:Try to run this test program as root.",
				strerror(errno));
		return CUE_SINIT_FAILED;
	}

	r = amdgpu_query_hw_ip_info(device_handle, AMDGPU_HW_IP_VCN_ENC, 0, &info);
	if (!r) {
		vcn_ip_version_major = info.hw_ip_version_major;
		vcn_ip_version_minor = info.hw_ip_version_minor;
	}

	if (amdgpu_device_deinitialize(device_handle))
		return CU_FALSE;

	vcn_unified_ring = true;

	return CU_TRUE;
}

int suite_vcn_userq_enc_tests_init(void)
{
	int r;

	r = amdgpu_device_initialize(drm_amdgpu[0], &major_version,
				     &minor_version, &device_handle);
	if (r) {
		if ((r == -EACCES) && (errno == EACCES))
			printf("\n\nError:%s. "
			       "Hint:Try to run this test program as root.",
				strerror(errno));
		return CUE_SINIT_FAILED;
	}

	return CUE_SUCCESS;
}

int suite_vcn_userq_enc_tests_clean(void)
{
	int r;

	r = amdgpu_device_deinitialize(device_handle);
	if (r == 0)
		return CUE_SUCCESS;
	else
		return CUE_SCLEAN_FAILED;
}


static void amdgpu_cs_sq_head(uint32_t *base, int *offset, bool enc)
{
	/* signature */
	*(base + (*offset)++) = 0x00000010;
	*(base + (*offset)++) = 0x30000002;
	ib_checksum = base + (*offset)++;
	ib_size_in_dw = base + (*offset)++;

	/* engine info */
	*(base + (*offset)++) = 0x00000010;
	*(base + (*offset)++) = 0x30000001;
	*(base + (*offset)++) = enc ? 2 : 3;
	*(base + (*offset)++) = 0x00000000;
}

static void amdgpu_cs_sq_ib_tail(uint32_t *end)
{
	uint32_t size_in_dw;
	uint32_t checksum = 0;

	/* if the pointers are invalid, no need to process */
	if (ib_checksum == NULL || ib_size_in_dw == NULL)
		return;

	size_in_dw = end - ib_size_in_dw - 1;
	*ib_size_in_dw = size_in_dw;
	*(ib_size_in_dw + 4) = size_in_dw * sizeof(uint32_t);

	for (int i = 0; i < size_in_dw; i++)
		checksum += *(ib_checksum + 2 + i);

	*ib_checksum = checksum;

	ib_checksum = NULL;
	ib_size_in_dw = NULL;
}

static int
amdgpu_bo_alloc_and_map_uq(amdgpu_device_handle dev, unsigned int size,
			   unsigned int alignment, unsigned int heap, uint64_t alloc_flags,
			   uint64_t mapping_flags, amdgpu_bo_handle *bo, void **cpu,
			   uint64_t *mc_address, amdgpu_va_handle *va_handle,
			   uint32_t timeline_syncobj_handle, uint64_t point)
{
	struct amdgpu_bo_alloc_request request = {};
	amdgpu_bo_handle buf_handle;
	amdgpu_va_handle handle;
	uint64_t vmc_addr;
	int r;

	request.alloc_size = size;
	request.phys_alignment = alignment;
	request.preferred_heap = heap;
	request.flags = alloc_flags;

	r = amdgpu_bo_alloc(dev, &request, &buf_handle);
	if (r)
		return r;

	r = amdgpu_va_range_alloc(dev,
				  amdgpu_gpu_va_range_general,
				  size, alignment, 0, &vmc_addr,
				  &handle, 0);
	if (r)
		goto error_va_alloc;

	r = amdgpu_bo_va_op_raw2(dev, buf_handle, 0,  ALIGN(size, getpagesize()), vmc_addr,
				   AMDGPU_VM_PAGE_READABLE |
				   AMDGPU_VM_PAGE_WRITEABLE |
				   AMDGPU_VM_PAGE_EXECUTABLE |
				   mapping_flags,
				   AMDGPU_VA_OP_MAP,
				   timeline_syncobj_handle,
				   point, 0, 0);
	if (r)
		goto error_va_map;

	r = amdgpu_bo_cpu_map(buf_handle, cpu);
	if (r)
		goto error_cpu_map;


	*bo = buf_handle;
	*mc_address = vmc_addr;
	*va_handle = handle;
	return 0;

 error_cpu_map:
	amdgpu_bo_cpu_unmap(buf_handle);
 error_va_map:
	amdgpu_bo_va_op(buf_handle, 0, size, vmc_addr, 0, AMDGPU_VA_OP_UNMAP);
 error_va_alloc:
	amdgpu_bo_free(buf_handle);
	return r;
}

static inline int
amdgpu_bo_unmap_and_free_uq(amdgpu_device_handle dev, amdgpu_bo_handle bo,
			    amdgpu_va_handle va_handle, uint64_t mc_addr, uint64_t size,
			    uint32_t timeline_syncobj_handle, uint64_t point,
			    uint64_t syncobj_handles_array, uint32_t num_syncobj_handles)
{
	amdgpu_bo_cpu_unmap(bo);
	amdgpu_bo_va_op_raw2(dev, bo, 0, size, mc_addr, 0, AMDGPU_VA_OP_UNMAP,
				  timeline_syncobj_handle, point,
				  syncobj_handles_array, num_syncobj_handles);
	amdgpu_va_range_free(va_handle);
	amdgpu_bo_free(bo);

	return 0;

}

static void
amdgpu_userqueue_get_bo_handle(amdgpu_bo_handle bo, uint32_t *bo_handle)
{
	*bo_handle =  bo->handle;
}

static int timeline_syncobj_wait(uint32_t timeline_syncobj_handle)
{
	uint64_t point, signaled_point;
	uint64_t timeout;
	struct timespec tp;
	uint32_t flags = DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED;
	int r;

	do {
		r = amdgpu_cs_syncobj_query2(device_handle, &timeline_syncobj_handle,
					     (uint64_t *)&point, 1, flags);
		if (r)
			return r;

		timeout = 0;
		clock_gettime(CLOCK_MONOTONIC, &tp);
		timeout = tp.tv_sec * 1000000000ULL + tp.tv_nsec;
		timeout += 100000000; //100 millisec

		r = amdgpu_cs_syncobj_timeline_wait(device_handle, &timeline_syncobj_handle,
						    (uint64_t *)&point, 1, timeout,
						    DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
						    DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT |
						    DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE,
						    NULL);

		if (r)
			return r;

		r = amdgpu_cs_syncobj_query(device_handle,
								    &timeline_syncobj_handle,
								    &signaled_point, 1);
		if (r)
			return r;

	} while (point != signaled_point);

	return r;
}

static void alloc_doorbell(struct amdgpu_vcn_bo *doorbell_bo,
			   unsigned int size, unsigned int domain)
{
	struct amdgpu_bo_alloc_request req = {0};
	amdgpu_bo_handle buf_handle;
	int r;

	req.alloc_size = ALIGN(size, PAGE_SIZE);
	req.preferred_heap = domain;
	r = amdgpu_bo_alloc(device_handle, &req, &buf_handle);
	CU_ASSERT_EQUAL(r, 0);

	doorbell_bo->handle = buf_handle;
	doorbell_bo->size = req.alloc_size;

	r = amdgpu_bo_cpu_map(doorbell_bo->handle,
			      (void **)&doorbell_bo->ptr);
	CU_ASSERT_EQUAL(r, 0);
}

static void amdgpu_vcn_userqueue_create(void)
{
	uint64_t gtt_flags = 0;
	int r;

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, USERMODE_QUEUE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&rb_base.handle, &rb_base.ptr,
				&rb_base.addr, &rb_base.va_handle,
				timeline_syncobj_handle, ++point);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, 8,
					ALIGNMENT,
					AMDGPU_GEM_DOMAIN_GTT,
					gtt_flags,
					AMDGPU_VM_MTYPE_UC,
					&wptr.handle, &wptr.ptr,
					&wptr.addr, &wptr.va_handle,
					timeline_syncobj_handle, ++point);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, 8,
					ALIGNMENT,
					AMDGPU_GEM_DOMAIN_VRAM,
					gtt_flags,
					AMDGPU_VM_MTYPE_UC,
					&rptr.handle, &rptr.ptr,
					&rptr.addr, &rptr.va_handle,
					timeline_syncobj_handle, ++point);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE,
				ALIGNMENT,
				AMDGPU_GEM_DOMAIN_VRAM,
				AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
				AMDGPU_VM_MTYPE_UC,
				&fence.handle, &fence.ptr,
				&fence.addr, &fence.va_handle,
				timeline_syncobj_handle, ++point);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&ib_ptr.handle, &ib_ptr.ptr,
				&ib_ptr.addr, &ib_ptr.va_handle,
				timeline_syncobj_handle, ++point);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	alloc_doorbell(&doorbell, PAGE_SIZE, AMDGPU_GEM_DOMAIN_DOORBELL);
	alloc_doorbell(&agdb_doorbell, PAGE_SIZE, AMDGPU_GEM_DOMAIN_DOORBELL);

	doorbell_ptr = (uint32_t *)doorbell.ptr;
	agdb_doorbell_ptr = (uint32_t *) agdb_doorbell.ptr;

	amdgpu_userqueue_get_bo_handle(doorbell.handle, &db_handle);
	amdgpu_userqueue_get_bo_handle(agdb_doorbell.handle, &agdb_handle);
	mqd.agdb_offset = VCN_DOORBELL_INDEX;
	mqd.agdb_handle = agdb_handle;

	/* Create the Usermode Queue */
	r = amdgpu_create_userqueue(device_handle, AMDGPU_HW_IP_VCN_ENC,
				    db_handle, VCN_DOORBELL_INDEX,
				    rb_base.addr, USERMODE_QUEUE_SIZE,
				    wptr.addr, rptr.addr, &mqd, 0,
				    (uint32_t *)&q_id);
	CU_ASSERT_EQUAL(r, 0);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle);
}

static void amdgpu_vcn_userqueue_destroy(void)
{
	uint32_t timeline_syncobj_handle_end;
	uint64_t point2 = 0;
	int r;

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle_end);
	CU_ASSERT_EQUAL(r, 0);

	/* Free the Usermode Queue */
	r = amdgpu_free_userqueue(device_handle, q_id);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, rb_base.handle,
				     rb_base.va_handle, rb_base.addr,
				     USERMODE_QUEUE_SIZE,
				     timeline_syncobj_handle_end, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, wptr.handle,
				     wptr.va_handle, wptr.addr,
				     8,
				     timeline_syncobj_handle_end, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, rptr.handle,
				     rptr.va_handle, rptr.addr,
				     8,
				     timeline_syncobj_handle_end, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, fence.handle,
				     fence.va_handle, fence.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle_end, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle_end);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_unmap(doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_free(doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle_end);
}

static void amdgpu_cs_vcn_enc_create(void)
{
	int len, r;
	uint32_t *p_task_size = NULL;
	uint32_t task_offset = 0, st_offset;
	uint32_t *st_size = NULL;
	unsigned int width = 160, height = 128, buf_size;
	uint32_t fw_maj = 1, fw_min = 9;
	uint32_t timeline_syncobj_handle_enc;
	uint64_t point_enc = 0;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	struct drm_amdgpu_userq_fence_info *fence_info = NULL;
	uint32_t *ref_val_lo, *ref_val_hi;
	uint32_t syncobj_handle, syncarray[2];
	uint64_t s_handle;
	uint64_t gpu_addr, reference_val;
	int syncobj_fd, i;
#if FENCE_LECAGY
	int test_pattern = 0xABADCAFE;
#endif

	if (vcn_ip_version_major == 2) {
		fw_maj = 1;
		fw_min = 1;
	} else if (vcn_ip_version_major == 3) {
		fw_maj = 1;
		fw_min = 0;
	}

	gWidth = width;
	gHeight = height;
	buf_size = ALIGN(width, 256) * ALIGN(height, 32) * 3 / 2;
	enc_task_id = 1;

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle_enc);
	CU_ASSERT_EQUAL(r, 0);

	amdgpu_vcn_userqueue_create();

	ib_cpu = (uint32_t *)ib_ptr.ptr;
	wptr_cpu = (uint32_t *)rb_base.ptr;
	memset(wptr_cpu, 0, USERMODE_QUEUE_SIZE);

	wptr_va_cpu = (uint32_t *)wptr.ptr;
	memset(wptr_va_cpu, 0, sizeof(*wptr_va_cpu));

	r = amdgpu_bo_cpu_map(rptr.handle, (void **)&rptr.ptr);
	CU_ASSERT_EQUAL(r, 0);

	rptr_va_cpu = (uint32_t *)rptr.ptr;
	rptr_va_cpu_64 = (uint64_t *)rptr.ptr;

	memset(rptr_va_cpu, 0, sizeof(*rptr_va_cpu));
	memset(rptr_va_cpu_64, 0, sizeof(*rptr_va_cpu_64));

	r = amdgpu_bo_alloc_and_map_uq(device_handle, ENCODE_BUF_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&enc_buf.handle, &enc_buf.ptr,
				&enc_buf.addr, &enc_buf.va_handle,
				timeline_syncobj_handle_enc, ++point_enc);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, buf_size * 2, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&cpb_buf.handle, &cpb_buf.ptr,
				&cpb_buf.addr, &cpb_buf.va_handle,
				timeline_syncobj_handle_enc, ++point_enc);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle_enc);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_map(enc_buf.handle, (void **)&enc_buf.ptr);
	memset(enc_buf.ptr, 0, 128 * 1024);
	r = amdgpu_bo_cpu_unmap(enc_buf.handle);

	r = amdgpu_bo_cpu_map(cpb_buf.handle, (void **)&cpb_buf.ptr);
	memset(cpb_buf.ptr, 0, buf_size * 2);
	r = amdgpu_bo_cpu_unmap(cpb_buf.handle);

	len = 0;
	if (vcn_unified_ring)
		amdgpu_cs_sq_head(ib_cpu, &len, true);

	/* session info */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000001;	/* RENCODE_IB_PARAM_SESSION_INFO */
	ib_cpu[len++] = ((fw_maj << 16) | (fw_min << 0));
	ib_cpu[len++] = enc_buf.addr >> 32;
	ib_cpu[len++] = enc_buf.addr;
	ib_cpu[len++] = 1;	/* RENCODE_ENGINE_TYPE_ENCODE; */
	*st_size = (len - st_offset) * 4;

	/* task info */
	task_offset = len;
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000002;	/* RENCODE_IB_PARAM_TASK_INFO */
	p_task_size = &ib_cpu[len++];
	ib_cpu[len++] = enc_task_id++;	/* task_id */
	ib_cpu[len++] = 0;	/* feedback */
	*st_size = (len - st_offset) * 4;

	/* op init */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x01000001;	/* RENCODE_IB_OP_INITIALIZE */
	*st_size = (len - st_offset) * 4;

	/* session_init */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000003;	/* RENCODE_IB_PARAM_SESSION_INIT */
	ib_cpu[len++] = 1;	/* RENCODE_ENCODE_STANDARD_H264 */
	ib_cpu[len++] = width;
	ib_cpu[len++] = height;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;	/* pre encode mode */
	ib_cpu[len++] = 0;	/* chroma enabled : false */
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;
	*st_size = (len - st_offset) * 4;

	/* slice control */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00200001;	/* RENCODE_H264_IB_PARAM_SLICE_CONTROL */
	ib_cpu[len++] = 0;	/* RENCODE_H264_SLICE_CONTROL_MODE_FIXED_MBS */
	ib_cpu[len++] = ALIGN(width, 16) / 16 * ALIGN(height, 16) / 16;
	*st_size = (len - st_offset) * 4;

	/* enc spec misc */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00200002;	/* RENCODE_H264_IB_PARAM_SPEC_MISC */
	ib_cpu[len++] = 0;	/* constrained intra pred flag */
	ib_cpu[len++] = 0;	/* cabac enable */
	ib_cpu[len++] = 0;	/* cabac init idc */
	ib_cpu[len++] = 1;	/* half pel enabled */
	ib_cpu[len++] = 1;	/* quarter pel enabled */
	ib_cpu[len++] = 100;	/* BASELINE profile */
	ib_cpu[len++] = 11;	/* level */
	if (vcn_ip_version_major >= 3) {
		ib_cpu[len++] = 0;	/* b_picture_enabled */
		ib_cpu[len++] = 0;	/* weighted_bipred_idc */
	}
	*st_size = (len - st_offset) * 4;

	/* deblocking filter */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00200004;	/* RENCODE_H264_IB_PARAM_DEBLOCKING_FILTER */
	ib_cpu[len++] = 0;	/* disable deblocking filter idc */
	ib_cpu[len++] = 0;	/* alpha c0 offset */
	ib_cpu[len++] = 0;	/* tc offset */
	ib_cpu[len++] = 0;	/* cb offset */
	ib_cpu[len++] = 0;	/* cr offset */
	*st_size = (len - st_offset) * 4;

	/* layer control */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000004;	/* RENCODE_IB_PARAM_LAYER_CONTROL */
	ib_cpu[len++] = 1;	/* max temporal layer */
	ib_cpu[len++] = 1;	/* no of temporal layer */
	*st_size = (len - st_offset) * 4;

	/* rc_session init */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000006;	/* RENCODE_IB_PARAM_RATE_CONTROL_SESSION_INIT */
	ib_cpu[len++] = 0;	/* rate control */
	ib_cpu[len++] = 48;	/* vbv buffer level */
	*st_size = (len - st_offset) * 4;

	/* quality params */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000009;	/* RENCODE_IB_PARAM_QUALITY_PARAMS */
	ib_cpu[len++] = 0;	/* vbaq mode */
	ib_cpu[len++] = 0;	/* scene change sensitivity */
	ib_cpu[len++] = 0;	/* scene change min idr interval */
	ib_cpu[len++] = 0;
	if (vcn_ip_version_major >= 3)
		ib_cpu[len++] = 0;
	*st_size = (len - st_offset) * 4;

	/* layer select */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000005;	/* RENCODE_IB_PARAM_LAYER_SELECT */
	ib_cpu[len++] = 0;	/* temporal layer */
	*st_size = (len - st_offset) * 4;

	/* rc layer init */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000007;	/* RENCODE_IB_PARAM_RATE_CONTROL_LAYER_INIT */
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 25;
	ib_cpu[len++] = 1;
	ib_cpu[len++] = 0x01312d00;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 0;
	*st_size = (len - st_offset) * 4;

	/* layer select */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000005;	/* RENCODE_IB_PARAM_LAYER_SELECT */
	ib_cpu[len++] = 0;	/* temporal layer */
	*st_size = (len - st_offset) * 4;

	/* rc per pic */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000008;	/* RENCODE_IB_PARAM_RATE_CONTROL_PER_PICTURE */
	ib_cpu[len++] = 20;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 51;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 1;
	ib_cpu[len++] = 0;
	ib_cpu[len++] = 1;
	ib_cpu[len++] = 0;
	*st_size = (len - st_offset) * 4;

	/* op init rc */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x01000004;	/* RENCODE_IB_OP_INIT_RC */
	*st_size = (len - st_offset) * 4;

	/* op init rc vbv */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x01000005;	/* RENCODE_IB_OP_INIT_RC_VBV_BUFFER_LEVEL */
	*st_size = (len - st_offset) * 4;

	*p_task_size = (len - task_offset) * 4;

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);

	/* add protected multifence  */
	r = drmSyncobjCreate(device_handle->fd, 0, &syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjHandleToFD(device_handle->fd, syncobj_handle, &syncobj_fd);
	CU_ASSERT_EQUAL(r, 0);

	syncarray[0] = syncobj_handle;
	s_handle = syncobj_handle;

	wptr_cpu[rb_len++] = VCN_HDP_FLUSH_CMD;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_IB;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = upper_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = len;
#if FENCE_LECAGY
	wptr_cpu[rb_len++] = VCN_ENC_CMD_FENCE;
	wptr_cpu[rb_len++] = lower_32_bits(fence.addr);
	wptr_cpu[rb_len++] = upper_32_bits(fence.addr);
	wptr_cpu[rb_len++] = test_pattern; // ABADCAFE
#endif

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_SIG_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	ref_val_lo = &wptr_cpu[rb_len++];
	ref_val_hi = &wptr_cpu[rb_len++];
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu =  ALIGN(rb_len, 16);
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);
	rb_len = *wptr_va_cpu;

	signal_data.queue_id = q_id;
	signal_data.syncobj_handles = (uintptr_t)syncarray;
	signal_data.num_syncobj_handles = 1;
	signal_data.bo_read_handles = 0;
	signal_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	signal_data.num_bo_read_handles = 0;
	signal_data.num_bo_write_handles = 1;

	r = amdgpu_userq_signal(device_handle, &signal_data);
	CU_ASSERT_EQUAL(r, 0);

	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	wait_data.waitq_id = q_id;
	wait_data.syncobj_handles = (uintptr_t)&s_handle;
	wait_data.syncobj_timeline_handles = 0;
	wait_data.syncobj_timeline_points = 0;
	wait_data.bo_read_handles = 0;
	wait_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	wait_data.num_syncobj_timeline_handles = 0;
	wait_data.num_syncobj_handles = 1;
	wait_data.num_bo_read_handles = 0;
	wait_data.num_bo_write_handles = 1;
	wait_data.out_fences = (uintptr_t)NULL;
	wait_data.num_fences = 0;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	fence_info = malloc(wait_data.num_fences * sizeof(struct drm_amdgpu_userq_fence_info));
	wait_data.out_fences = (uintptr_t)fence_info;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	gpu_addr = fence_info->va;
	reference_val = fence_info->value;

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_WAIT_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = upper_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = lower_32_bits(reference_val);
	wptr_cpu[rb_len++] = upper_32_bits(reference_val);
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu = ALIGN(rb_len, 16);
	rb_len = *wptr_va_cpu;
	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

#if FENCE_LECAGY
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in encode create test\n");
			break;
		}
		count++;
	}

	CU_ASSERT_EQUAL(*fence_cpu, test_pattern);

#endif

	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			break;
	}

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle_enc);
}

static void amdgpu_cs_vcn_enc_encode_frame(int frame_type)
{
	struct amdgpu_vcn_bo bs_buf, fb_buf, input_buf;
	uint32_t timeline_syncobj_handle_enc_frame;
	uint32_t timeline_syncobj_handle_frame;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	struct drm_amdgpu_userq_fence_info *fence_info = NULL;
	uint32_t syncobj_handle, syncarray[2];
	uint32_t *ref_val_lo, *ref_val_hi;
	uint64_t s_handle;
	uint64_t gpu_addr, reference_val;
	int syncobj_fd, i;
	uint64_t point_enc_frame = 0, point_frame = 0;
	int len, r;
	unsigned int width = 160, height = 128, buf_size;
	uint32_t *p_task_size = NULL;
	uint32_t task_offset = 0, st_offset;
	uint32_t *st_size = NULL;
	uint32_t fw_maj = 1, fw_min = 9;

	if (vcn_ip_version_major == 2) {
		fw_maj = 1;
		fw_min = 1;
	} else if (vcn_ip_version_major == 3) {
		fw_maj = 1;
		fw_min = 0;
	}

	gSliceType = frame_type;
	buf_size = ALIGN(width, 256) * ALIGN(height, 32) * 3 / 2;
	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle_enc_frame);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&bs_buf.handle, &bs_buf.ptr,
				&bs_buf.addr, &bs_buf.va_handle,
				timeline_syncobj_handle_enc_frame, ++point_enc_frame);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&fb_buf.handle, &fb_buf.ptr,
				&fb_buf.addr, &fb_buf.va_handle,
				timeline_syncobj_handle_enc_frame, ++point_enc_frame);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, buf_size, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				0,
				AMDGPU_VM_MTYPE_UC,
				&input_buf.handle, &input_buf.ptr,
				&input_buf.addr, &input_buf.va_handle,
				timeline_syncobj_handle_enc_frame, ++point_enc_frame);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle_enc_frame);
	CU_ASSERT_EQUAL(r, 0);

	memset(bs_buf.ptr, 0, 4096);
	memset(fb_buf.ptr, 0, 4096);
	for (int i = 0; i < ALIGN(height, 32) * 3 / 2; i++)
		memcpy(input_buf.ptr + i * ALIGN(width, 256), frame + i * width, width);

	len = 0;
	if (vcn_unified_ring)
		amdgpu_cs_sq_head(ib_cpu, &len, true);

	/* session info */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000001;	/* RENCODE_IB_PARAM_SESSION_INFO */
	ib_cpu[len++] = ((fw_maj << 16) | (fw_min << 0));
	ib_cpu[len++] = enc_buf.addr >> 32;
	ib_cpu[len++] = enc_buf.addr;
	ib_cpu[len++] = 1;	/* RENCODE_ENGINE_TYPE_ENCODE */;
	*st_size = (len - st_offset) * 4;

	/* task info */
	task_offset = len;
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000002;	/* RENCODE_IB_PARAM_TASK_INFO */
	p_task_size = &ib_cpu[len++];
	ib_cpu[len++] = enc_task_id++;	/* task_id */
	ib_cpu[len++] = 1;	/* feedback */
	*st_size = (len - st_offset) * 4;

	if (frame_type == 2) {
		/* sps */
		st_offset = len;
		st_size = &ib_cpu[len++];	/* size */
		if (vcn_ip_version_major == 1)
			ib_cpu[len++] = 0x00000020;	/* IB_PARAM_DIRECT_OUTPUT_NALU vcn 1 */
		else
			ib_cpu[len++] = 0x0000000a;	/* IB_PARAM_DIRECT_OUTPUT_NALU other vcn */
		ib_cpu[len++] = 0x00000002;	/* RENCODE_DIRECT_OUTPUT_NALU_TYPE_SPS */
		ib_cpu[len++] = 0x00000011;	/* sps len */
		ib_cpu[len++] = 0x00000001;	/* start code */
		ib_cpu[len++] = 0x6764440b;
		ib_cpu[len++] = 0xac54c284;
		ib_cpu[len++] = 0x68078442;
		ib_cpu[len++] = 0x37000000;
		*st_size = (len - st_offset) * 4;

		/* pps */
		st_offset = len;
		st_size = &ib_cpu[len++];	/* size */
		if (vcn_ip_version_major == 1)
			ib_cpu[len++] = 0x00000020;	/* IB_PARAM_DIRECT_OUTPUT_NALU vcn 1*/
		else
			ib_cpu[len++] = 0x0000000a;	/* IB_PARAM_DIRECT_OUTPUT_NALU other vcn*/
		ib_cpu[len++] = 0x00000003;	/* RENCODE_DIRECT_OUTPUT_NALU_TYPE_PPS */
		ib_cpu[len++] = 0x00000008;	/* pps len */
		ib_cpu[len++] = 0x00000001;	/* start code */
		ib_cpu[len++] = 0x68ce3c80;
		*st_size = (len - st_offset) * 4;
	}

	/* slice header */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x0000000a; /* RENCODE_IB_PARAM_SLICE_HEADER vcn 1 */
	else
		ib_cpu[len++] = 0x0000000b; /* RENCODE_IB_PARAM_SLICE_HEADER other vcn */
	if (frame_type == 2) {
		ib_cpu[len++] = 0x65000000;
		ib_cpu[len++] = 0x11040000;
	} else {
		ib_cpu[len++] = 0x41000000;
		ib_cpu[len++] = 0x34210000;
	}
	ib_cpu[len++] = 0xe0000000;
	amdgpu_cs_vcn_ib_zero_count(&len, 13);

	ib_cpu[len++] = 0x00000001;
	ib_cpu[len++] = 0x00000008;
	ib_cpu[len++] = 0x00020000;
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = 0x00000001;
	ib_cpu[len++] = 0x00000015;
	ib_cpu[len++] = 0x00020001;
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = 0x00000001;
	ib_cpu[len++] = 0x00000003;
	amdgpu_cs_vcn_ib_zero_count(&len, 22);
	*st_size = (len - st_offset) * 4;

	/* encode params */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x0000000b;	/* RENCODE_IB_PARAM_ENCODE_PARAMS vcn 1 */
	else
		ib_cpu[len++] = 0x0000000f;	/* RENCODE_IB_PARAM_ENCODE_PARAMS other vcn */
	ib_cpu[len++] = frame_type;
	ib_cpu[len++] = 0x0001f000;
	ib_cpu[len++] = input_buf.addr >> 32;
	ib_cpu[len++] = input_buf.addr;
	ib_cpu[len++] = (input_buf.addr + ALIGN(width, 256) * ALIGN(height, 32)) >> 32;
	ib_cpu[len++] = input_buf.addr + ALIGN(width, 256) * ALIGN(height, 32);
	ib_cpu[len++] = 0x00000100;
	ib_cpu[len++] = 0x00000080;
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = 0xffffffff;
	ib_cpu[len++] = 0x00000000;
	*st_size = (len - st_offset) * 4;

	/* encode params h264 */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00200003;	/* RENCODE_H264_IB_PARAM_ENCODE_PARAMS */
	if (vcn_ip_version_major <= 2) {
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0xffffffff;
	} else {
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0xffffffff;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0xffffffff;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000001;
	}
	*st_size = (len - st_offset) * 4;

	/* encode context */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x0000000d;	/* ENCODE_CONTEXT_BUFFER  vcn 1 */
	else
		ib_cpu[len++] = 0x00000011;	/* ENCODE_CONTEXT_BUFFER  other vcn */
	ib_cpu[len++] = cpb_buf.addr >> 32;
	ib_cpu[len++] = cpb_buf.addr;
	ib_cpu[len++] = 0x00000000;	/* swizzle mode */
	ib_cpu[len++] = 0x00000100;	/* luma pitch */
	ib_cpu[len++] = 0x00000100;	/* chroma pitch */
	ib_cpu[len++] = 0x00000002; /* no reconstructed picture */
	ib_cpu[len++] = 0x00000000;	/* reconstructed pic 1 luma offset */
	ib_cpu[len++] = ALIGN(width, 256) * ALIGN(height, 32);	/* pic1 chroma offset */
	if (vcn_ip_version_major == 4)
		amdgpu_cs_vcn_ib_zero_count(&len, 2);
	ib_cpu[len++] = ALIGN(width, 256) * ALIGN(height, 32) * 3 / 2;	/* pic2 luma offset */
	ib_cpu[len++] = ALIGN(width, 256) * ALIGN(height, 32) * 5 / 2;	/* pic2 chroma offset */

	amdgpu_cs_vcn_ib_zero_count(&len, 280);
	*st_size = (len - st_offset) * 4;

	/* bitstream buffer */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x0000000e;	/* VIDEO_BITSTREAM_BUFFER vcn 1 */
	else
		ib_cpu[len++] = 0x00000012;	/* VIDEO_BITSTREAM_BUFFER other vcn */

	ib_cpu[len++] = 0x00000000;	/* mode */
	ib_cpu[len++] = bs_buf.addr >> 32;
	ib_cpu[len++] = bs_buf.addr;
	ib_cpu[len++] = 0x0001f000;
	ib_cpu[len++] = 0x00000000;
	*st_size = (len - st_offset) * 4;

	/* feedback */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x00000010;	/* FEEDBACK_BUFFER vcn 1 */
	else
		ib_cpu[len++] = 0x00000015;	/* FEEDBACK_BUFFER vcn 2,3 */
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = fb_buf.addr >> 32;
	ib_cpu[len++] = fb_buf.addr;
	ib_cpu[len++] = 0x00000010;
	ib_cpu[len++] = 0x00000028;
	*st_size = (len - st_offset) * 4;

	/* intra refresh */
	st_offset = len;
	st_size = &ib_cpu[len++];
	if (vcn_ip_version_major == 1)
		ib_cpu[len++] = 0x0000000c;	/* INTRA_REFRESH vcn 1 */
	else
		ib_cpu[len++] = 0x00000010;	/* INTRA_REFRESH vcn 2,3 */
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = 0x00000000;
	ib_cpu[len++] = 0x00000000;
	*st_size = (len - st_offset) * 4;

	if (vcn_ip_version_major != 1) {
		/* Input Format */
		st_offset = len;
		st_size = &ib_cpu[len++];
		ib_cpu[len++] = 0x0000000c;
		ib_cpu[len++] = 0x00000000;	/* RENCODE_COLOR_VOLUME_G22_BT709 */
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;	/* RENCODE_COLOR_BIT_DEPTH_8_BIT */
		ib_cpu[len++] = 0x00000000;	/* RENCODE_COLOR_PACKING_FORMAT_NV12 */
		*st_size = (len - st_offset) * 4;

		/* Output Format */
		st_offset = len;
		st_size = &ib_cpu[len++];
		ib_cpu[len++] = 0x0000000d;
		ib_cpu[len++] = 0x00000000;	/* RENCODE_COLOR_VOLUME_G22_BT709 */
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;
		ib_cpu[len++] = 0x00000000;	/* RENCODE_COLOR_BIT_DEPTH_8_BIT */
		*st_size = (len - st_offset) * 4;
	}
	/* op_speed */
	st_offset = len;
	st_size = &ib_cpu[len++];
	ib_cpu[len++] = 0x01000006;	/* SPEED_ENCODING_MODE */
	*st_size = (len - st_offset) * 4;

	/* op_enc */
	st_offset = len;
	st_size = &ib_cpu[len++];
	ib_cpu[len++] = 0x01000003;
	*st_size = (len - st_offset) * 4;

	*p_task_size = (len - task_offset) * 4;

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);

	/* add protected multifence */
	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle_frame);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjCreate(device_handle->fd, 0, &syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjHandleToFD(device_handle->fd, syncobj_handle, &syncobj_fd);
	CU_ASSERT_EQUAL(r, 0);

	syncarray[0] = syncobj_handle;
	s_handle = syncobj_handle;

	wptr_cpu[rb_len++] = VCN_HDP_FLUSH_CMD;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_IB;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = upper_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = len;
#if FENCE_LECAGY
	wptr_cpu[rb_len++] = VCN_ENC_CMD_FENCE;
	wptr_cpu[rb_len++] = lower_32_bits(fence.addr);
	wptr_cpu[rb_len++] = upper_32_bits(fence.addr);
	wptr_cpu[rb_len++] = test_pattern; // ABADCAFE
#endif

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_SIG_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	ref_val_lo = &wptr_cpu[rb_len++];
	ref_val_hi = &wptr_cpu[rb_len++];
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu =  ALIGN(rb_len, 16);
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);
	rb_len = *wptr_va_cpu;

	signal_data.queue_id = q_id;
	signal_data.syncobj_handles = (uintptr_t)syncarray;
	signal_data.num_syncobj_handles = 1;
	signal_data.bo_read_handles = 0;
	signal_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	signal_data.num_bo_read_handles = 0;
	signal_data.num_bo_write_handles = 1;

	r = amdgpu_userq_signal(device_handle, &signal_data);
	CU_ASSERT_EQUAL(r, 0);

	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	wait_data.waitq_id = q_id;
	wait_data.syncobj_handles = (uintptr_t)&s_handle;
	wait_data.syncobj_timeline_handles = 0;
	wait_data.syncobj_timeline_points = 0;
	wait_data.bo_read_handles = 0;
	wait_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	wait_data.num_syncobj_timeline_handles = 0;
	wait_data.num_syncobj_handles = 1;
	wait_data.num_bo_read_handles = 0;
	wait_data.num_bo_write_handles = 1;
	wait_data.out_fences = (uintptr_t)NULL;
	wait_data.num_fences = 0;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	fence_info = malloc(wait_data.num_fences * sizeof(struct drm_amdgpu_userq_fence_info));
	wait_data.out_fences = (uintptr_t)fence_info;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	gpu_addr = fence_info->va;
	reference_val = fence_info->value;

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_WAIT_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = upper_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = lower_32_bits(reference_val);
	wptr_cpu[rb_len++] = upper_32_bits(reference_val);
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu = ALIGN(rb_len, 16);
	rb_len = *wptr_va_cpu;
	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

#if FENCE_LECAGY
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in encode create test\n");
			break;
		}
		count++;
	}

	CU_ASSERT_EQUAL(*fence_cpu, test_pattern);
#endif

	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			break;
	}
	/* check result */
	check_result(fb_buf, bs_buf, frame_type);
	r = amdgpu_bo_unmap_and_free_uq(device_handle, fb_buf.handle,
				     fb_buf.va_handle, fb_buf.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle_frame, ++point_frame,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, bs_buf.handle,
				     bs_buf.va_handle, bs_buf.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle_frame, ++point_frame,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, input_buf.handle,
				     input_buf.va_handle, input_buf.addr,
				     buf_size,
				     timeline_syncobj_handle_frame, ++point_frame,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle_frame);
	CU_ASSERT_EQUAL(r, 0);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle_enc_frame);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle_frame);
}

static void amdgpu_cs_vcn_enc_encode(void)
{
	amdgpu_cs_vcn_enc_encode_frame(2);	/* IDR frame */
}

static void amdgpu_cs_vcn_enc_destroy(void)
{
	int len = 0, r;
	uint32_t *p_task_size = NULL;
	uint32_t task_offset = 0, st_offset;
	uint32_t *st_size = NULL;
	uint32_t fw_maj = 1, fw_min = 9;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	struct drm_amdgpu_userq_fence_info *fence_info = NULL;
	uint32_t syncobj_handle, syncarray[2];
	uint64_t s_handle;
	uint64_t gpu_addr, reference_val;
	uint32_t timeline_syncobj_handle_end;
	uint64_t point_end = 0;
	uint32_t *ref_val_lo, *ref_val_hi;
	int syncobj_fd, i;

#if FENCE_LECAGY
	int test_pattern = 0xBAD0CAFE;
#endif

	if (vcn_ip_version_major == 2) {
		fw_maj = 1;
		fw_min = 1;
	} else if (vcn_ip_version_major == 3) {
		fw_maj = 1;
		fw_min = 0;
	}

	if (vcn_unified_ring)
		amdgpu_cs_sq_head(ib_cpu, &len, true);


	/* session info */
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000001;	/* RENCODE_IB_PARAM_SESSION_INFO */
	ib_cpu[len++] = ((fw_maj << 16) | (fw_min << 0));
	ib_cpu[len++] = enc_buf.addr >> 32;
	ib_cpu[len++] = enc_buf.addr;
	ib_cpu[len++] = 1;	/* RENCODE_ENGINE_TYPE_ENCODE; */
	*st_size = (len - st_offset) * 4;

	/* task info */
	task_offset = len;
	st_offset = len;
	st_size = &ib_cpu[len++];	/* size */
	ib_cpu[len++] = 0x00000002;	/* RENCODE_IB_PARAM_TASK_INFO */
	p_task_size = &ib_cpu[len++];
	ib_cpu[len++] = enc_task_id++;	/* task_id */
	ib_cpu[len++] = 0;	/* feedback */
	*st_size = (len - st_offset) * 4;

	/*  op close */
	st_offset = len;
	st_size = &ib_cpu[len++];
	ib_cpu[len++] = 0x01000002;	/* RENCODE_IB_OP_CLOSE_SESSION */
	*st_size = (len - st_offset) * 4;

	*p_task_size = (len - task_offset) * 4;

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);

	/* add protected multifence */
	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle_end);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjCreate(device_handle->fd, 0, &syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjHandleToFD(device_handle->fd, syncobj_handle, &syncobj_fd);
	CU_ASSERT_EQUAL(r, 0);

	syncarray[0] = syncobj_handle;
	s_handle = syncobj_handle;

	wptr_cpu[rb_len++] = VCN_HDP_FLUSH_CMD;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_IB;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = upper_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = len;
#if FENCE_LECAGY
	wptr_cpu[rb_len++] = VCN_ENC_CMD_FENCE;
	wptr_cpu[rb_len++] = lower_32_bits(fence.addr);
	wptr_cpu[rb_len++] = upper_32_bits(fence.addr);
	wptr_cpu[rb_len++] = test_pattern; // ABADCAFE
#endif

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_SIG_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	ref_val_lo = &wptr_cpu[rb_len++];
	ref_val_hi = &wptr_cpu[rb_len++];
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu =  ALIGN(rb_len, 16);
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);
	rb_len = *wptr_va_cpu;

	signal_data.queue_id = q_id;
	signal_data.syncobj_handles = (uintptr_t)syncarray;
	signal_data.num_syncobj_handles = 1;
	signal_data.bo_read_handles = 0;
	signal_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	signal_data.num_bo_read_handles = 0;
	signal_data.num_bo_write_handles = 1;

	r = amdgpu_userq_signal(device_handle, &signal_data);
	CU_ASSERT_EQUAL(r, 0);

	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	wait_data.waitq_id = q_id;
	wait_data.syncobj_handles = (uintptr_t)&s_handle;
	wait_data.syncobj_timeline_handles = 0;
	wait_data.syncobj_timeline_points = 0;
	wait_data.bo_read_handles = 0;
	wait_data.bo_write_handles = (uintptr_t)&ib_ptr.handle->handle;
	wait_data.num_syncobj_timeline_handles = 0;
	wait_data.num_syncobj_handles = 1;
	wait_data.num_bo_read_handles = 0;
	wait_data.num_bo_write_handles = 1;
	wait_data.out_fences = (uintptr_t)NULL;
	wait_data.num_fences = 0;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	fence_info = malloc(wait_data.num_fences * sizeof(struct drm_amdgpu_userq_fence_info));
	wait_data.out_fences = (uintptr_t)fence_info;

	r = amdgpu_userq_wait(device_handle, &wait_data);
	CU_ASSERT_EQUAL(r, 0);

	gpu_addr = fence_info->va;
	reference_val = fence_info->value;

	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_WAIT_CMD;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = upper_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = lower_32_bits(reference_val);
	wptr_cpu[rb_len++] = upper_32_bits(reference_val);
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu = ALIGN(rb_len, 16);
	rb_len = *wptr_va_cpu;
	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

#if FENCE_LECAGY
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in encode create test\n");
			break;
		}
		count++;
	}

	CU_ASSERT_EQUAL(*fence_cpu, test_pattern);
#endif
	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			break;
	}

	amdgpu_vcn_userqueue_destroy();

	r = amdgpu_bo_unmap_and_free_uq(device_handle, cpb_buf.handle,
				     cpb_buf.va_handle, cpb_buf.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle_end, ++point_end,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, enc_buf.handle,
				     enc_buf.va_handle, enc_buf.addr,
				     32 * PAGE_SIZE,
				     timeline_syncobj_handle_end, ++point_end,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle_end);
	CU_ASSERT_EQUAL(r, 0);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle_end);
}
