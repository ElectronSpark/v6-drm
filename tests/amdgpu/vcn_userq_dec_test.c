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

#include "decode_messages.h"
#include "vcn_tests.h"

#define upper_32_bits(n) ((__u32)(((n) >> 16) >> 16))
#define lower_32_bits(n) ((__u32)(n))

#define PAGE_SIZE				4096
#define USERMODE_QUEUE_SIZE			(PAGE_SIZE * 32)
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

#define DECODE_CMD_MSG_BUFFER				0x00000000
#define DECODE_CMD_DPB_BUFFER				0x00000001
#define DECODE_CMD_DECODING_TARGET_BUFFER		0x00000002
#define DECODE_CMD_FEEDBACK_BUFFER			0x00000003
#define DECODE_CMD_PROB_TBL_BUFFER			0x00000004
#define DECODE_CMD_SESSION_CONTEXT_BUFFER		0x00000005
#define DECODE_CMD_BITSTREAM_BUFFER			0x00000100
#define DECODE_CMD_IT_SCALING_TABLE_BUFFER		0x00000204
#define DECODE_CMD_CONTEXT_BUFFER			0x00000206

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
static struct  amdgpu_vcn_bo csa;
static uint32_t major_version;
static uint32_t minor_version;
static uint32_t q_id, db_handle, agdb_handle;
static struct drm_amdgpu_userq_mqd_vcn mqd;
static uint32_t *ib_checksum;
static uint32_t *ib_size_in_dw;
static uint32_t *ib_cpu, *wptr_cpu, *fence_cpu;
static uint32_t *wptr_va_cpu, *rptr_va_cpu;
static uint32_t *doorbell_ptr, *agdb_doorbell_ptr;
static uint64_t *rptr_va_cpu_64;
static uint32_t rb_len;

static bool vcn_dec_sw_ring = false;
static bool vcn_unified_ring = false;
static uint32_t vcn_ip_version_major;
static uint32_t vcn_ip_version_minor;

typedef struct rvcn_decode_buffer_s {
	unsigned int valid_buf_flag;
	unsigned int msg_buffer_address_hi;
	unsigned int msg_buffer_address_lo;
	unsigned int dpb_buffer_address_hi;
	unsigned int dpb_buffer_address_lo;
	unsigned int target_buffer_address_hi;
	unsigned int target_buffer_address_lo;
	unsigned int session_contex_buffer_address_hi;
	unsigned int session_contex_buffer_address_lo;
	unsigned int bitstream_buffer_address_hi;
	unsigned int bitstream_buffer_address_lo;
	unsigned int context_buffer_address_hi;
	unsigned int context_buffer_address_lo;
	unsigned int feedback_buffer_address_hi;
	unsigned int feedback_buffer_address_lo;
	unsigned int luma_hist_buffer_address_hi;
	unsigned int luma_hist_buffer_address_lo;
	unsigned int prob_tbl_buffer_address_hi;
	unsigned int prob_tbl_buffer_address_lo;
	unsigned int sclr_coeff_buffer_address_hi;
	unsigned int sclr_coeff_buffer_address_lo;
	unsigned int it_sclr_table_buffer_address_hi;
	unsigned int it_sclr_table_buffer_address_lo;
	unsigned int sclr_target_buffer_address_hi;
	unsigned int sclr_target_buffer_address_lo;
	unsigned int cenc_size_info_buffer_address_hi;
	unsigned int cenc_size_info_buffer_address_lo;
	unsigned int mpeg2_pic_param_buffer_address_hi;
	unsigned int mpeg2_pic_param_buffer_address_lo;
	unsigned int mpeg2_mb_control_buffer_address_hi;
	unsigned int mpeg2_mb_control_buffer_address_lo;
	unsigned int mpeg2_idct_coeff_buffer_address_hi;
	unsigned int mpeg2_idct_coeff_buffer_address_lo;
} rvcn_decode_buffer_t;

typedef struct rvcn_decode_ib_package_s {
	unsigned int package_size;
	unsigned int package_type;
} rvcn_decode_ib_package_t;

static rvcn_decode_buffer_t *decode_buffer;
static struct amdgpu_vcn_bo session_ctx_buf;

static void amdgpu_vcn_userqueue_create(void);
static void amdgpu_vcn_userqueue_destroy(void);
static void amdgpu_cs_vcn_dec_create(void);
static void amdgpu_cs_vcn_dec_decode(void);
static void amdgpu_cs_vcn_dec_destroy(void);

CU_TestInfo userq_vcn_dec_tests[] = {
	{"VCN DEC create", amdgpu_cs_vcn_dec_create},
	{"VCN DEC decode", amdgpu_cs_vcn_dec_decode},
	{"VCN DEC destroy", amdgpu_cs_vcn_dec_destroy},
	CU_TEST_INFO_NULL,
};

CU_BOOL suite_vcn_userq_dec_tests_enable(void)
{
	struct drm_amdgpu_info_hw_ip info;
	int r;

	r = amdgpu_device_initialize(drm_amdgpu[0], &major_version,
				     &minor_version, &device_handle);
	if (r) {
		if ((r == -EACCES) && (errno == EACCES))
			printf("\nError:%s. "
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
	vcn_dec_sw_ring = true;

	return CU_TRUE;
}

int suite_vcn_userq_dec_tests_init(void)
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

int suite_vcn_userq_dec_tests_clean(void)
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
	uint32_t flags = DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED;
	uint64_t point, signaled_point;
	uint64_t timeout;
	struct timespec tp;
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

		r = amdgpu_cs_syncobj_query(device_handle, &timeline_syncobj_handle,
								    &signaled_point, 1);
		if (r)
			return r;

	} while (point != signaled_point);

	return r;
}

static void vcn_dec_cmd(uint64_t addr, unsigned int cmd, int *idx)
{
	/* Support decode software ring message */
	if (!(*idx)) {
		rvcn_decode_ib_package_t *ib_header;

		if (vcn_unified_ring)
			amdgpu_cs_sq_head(ib_cpu, idx, false);

		ib_header = (rvcn_decode_ib_package_t *)&ib_cpu[*idx];
		ib_header->package_size = sizeof(struct rvcn_decode_buffer_s) +
			sizeof(struct rvcn_decode_ib_package_s);

		(*idx)++;
		ib_header->package_type = (DECODE_IB_PARAM_DECODE_BUFFER);
		(*idx)++;

		decode_buffer = (rvcn_decode_buffer_t *)&(ib_cpu[*idx]);
		*idx += sizeof(struct rvcn_decode_buffer_s) / 4;
		memset(decode_buffer, 0, sizeof(struct rvcn_decode_buffer_s));
	}

	switch (cmd) {
	case DECODE_CMD_MSG_BUFFER:
		decode_buffer->valid_buf_flag |= DECODE_CMDBUF_FLAGS_MSG_BUFFER;
		decode_buffer->msg_buffer_address_hi = (addr >> 32);
		decode_buffer->msg_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_DPB_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_DPB_BUFFER);
		decode_buffer->dpb_buffer_address_hi = (addr >> 32);
		decode_buffer->dpb_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_DECODING_TARGET_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_DECODING_TARGET_BUFFER);
		decode_buffer->target_buffer_address_hi = (addr >> 32);
		decode_buffer->target_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_FEEDBACK_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_FEEDBACK_BUFFER);
		decode_buffer->feedback_buffer_address_hi = (addr >> 32);
		decode_buffer->feedback_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_PROB_TBL_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_PROB_TBL_BUFFER);
		decode_buffer->prob_tbl_buffer_address_hi = (addr >> 32);
		decode_buffer->prob_tbl_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_SESSION_CONTEXT_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_SESSION_CONTEXT_BUFFER);
		decode_buffer->session_contex_buffer_address_hi = (addr >> 32);
		decode_buffer->session_contex_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_BITSTREAM_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_BITSTREAM_BUFFER);
		decode_buffer->bitstream_buffer_address_hi = (addr >> 32);
		decode_buffer->bitstream_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_IT_SCALING_TABLE_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_IT_SCALING_BUFFER);
		decode_buffer->it_sclr_table_buffer_address_hi = (addr >> 32);
		decode_buffer->it_sclr_table_buffer_address_lo = (addr);
	break;
	case DECODE_CMD_CONTEXT_BUFFER:
		decode_buffer->valid_buf_flag |= (DECODE_CMDBUF_FLAGS_CONTEXT_BUFFER);
		decode_buffer->context_buffer_address_hi = (addr >> 32);
		decode_buffer->context_buffer_address_lo = (addr);
	break;
	default:
		printf("Not Support!\n");
	}
}

static void alloc_doorbell(struct amdgpu_vcn_bo *doorbell_bo,
			   unsigned size, unsigned domain)
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
	uint32_t timeline_syncobj_handle;
	uint64_t point = 0;
	int r;

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle);
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

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&csa.handle, &csa.ptr,
				&csa.addr, &csa.va_handle,
				timeline_syncobj_handle, ++point);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, USERMODE_QUEUE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_VRAM,
				AMDGPU_GEM_CREATE_VM_ALWAYS_VALID,
				AMDGPU_VM_MTYPE_UC,
				&rb_base.handle, &rb_base.ptr,
				&rb_base.addr, &rb_base.va_handle,
				timeline_syncobj_handle, ++point);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE,
				ALIGNMENT,
				AMDGPU_GEM_DOMAIN_VRAM,
				AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
				AMDGPU_VM_MTYPE_UC,
				&fence.handle, &fence.ptr,
				&fence.addr, &fence.va_handle,
				timeline_syncobj_handle, ++point);

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
				    (uint32_t *) &q_id);
	CU_ASSERT_EQUAL(r, 0);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle);
}

static void amdgpu_vcn_userqueue_destroy(void)
{
	uint32_t timeline_syncobj_handle;
	uint64_t point2 = 0;
	int r;

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	/* Free the Usermode Queue */
	r = amdgpu_free_userqueue(device_handle, q_id);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, csa.handle,
				     csa.va_handle, csa.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, rb_base.handle,
				     rb_base.va_handle, rb_base.addr,
				     USERMODE_QUEUE_SIZE,
				     timeline_syncobj_handle, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, wptr.handle,
				     wptr.va_handle, wptr.addr,
				     8,
				     timeline_syncobj_handle, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, rptr.handle,
				     rptr.va_handle, rptr.addr,
				     8,
				     timeline_syncobj_handle, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, fence.handle,
				     fence.va_handle, fence.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle, ++point2,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_unmap(doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_unmap(agdb_doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_free(doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_free(agdb_doorbell.handle);
	CU_ASSERT_EQUAL(r, 0);

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle);
}

static void amdgpu_cs_vcn_dec_create(void)
{
#if FENCE_LECAGY
	static int count;
	int test_pattern = 0xABADCAFE;
#endif
	uint32_t timeline_syncobj_handle, free_timeline_syncobj_handle;
	struct drm_amdgpu_userq_fence_info *fence_info;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	struct amdgpu_vcn_bo msg_buf;
	uint64_t gtt_flags, point3, point5;
	uint64_t gpu_addr, reference_val;
	uint64_t s_handle;
	uint32_t syncobj_handle, syncarray[2];
	uint32_t *ref_val_lo, *ref_val_hi;
	int syncobj_fd, i;
	int len, r;

	fence_info = NULL;
	gtt_flags = point3 = point5 = 0;
	amdgpu_vcn_userqueue_create();

	wptr_va_cpu = (uint32_t *)wptr.ptr;
	memset(wptr_va_cpu, 0, sizeof(*wptr_va_cpu));

	r = amdgpu_bo_cpu_map(rptr.handle, (void **)&rptr.ptr);
	CU_ASSERT_EQUAL(r, 0);

	rptr_va_cpu = (uint32_t *)rptr.ptr;
	rptr_va_cpu_64 = (uint64_t *)rptr.ptr;

	memset(rptr_va_cpu, 0, sizeof(*rptr_va_cpu));
	memset(rptr_va_cpu_64, 0, sizeof(*rptr_va_cpu_64));

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjCreate(device_handle->fd, 0, &free_timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&ib_ptr.handle, &ib_ptr.ptr,
				&ib_ptr.addr, &ib_ptr.va_handle,
				timeline_syncobj_handle, ++point3);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&msg_buf.handle, &msg_buf.ptr,
				&msg_buf.addr, &msg_buf.va_handle,
				timeline_syncobj_handle, ++point3);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle,  32 * PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT | AMDGPU_GEM_DOMAIN_VRAM,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&session_ctx_buf.handle, &session_ctx_buf.ptr,
				&session_ctx_buf.addr, &session_ctx_buf.va_handle,
				timeline_syncobj_handle, ++point3);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	ib_cpu = (uint32_t *)ib_ptr.ptr;

	memset(msg_buf.ptr, 0, 4096);
	memcpy(msg_buf.ptr, vcn_dec_create_msg, sizeof(vcn_dec_create_msg));

	rb_len = len = 0;
	vcn_dec_cmd(session_ctx_buf.addr, 5, &len);
	if (vcn_dec_sw_ring == true)
		vcn_dec_cmd(msg_buf.addr, 0, &len);

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);

	wptr_cpu = (uint32_t *)rb_base.ptr;
	fence_cpu = (uint32_t *)fence.ptr;
	memset(wptr_cpu, 0, USERMODE_QUEUE_SIZE);
	memset(fence_cpu, 0, PAGE_SIZE);


	/* add protected multifence */
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
	wptr_cpu[rb_len++] = test_pattern; /* ABADCAFE */
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
	rb_len = *wptr_va_cpu;
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);

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
	rb_len++;
	wptr_cpu[rb_len++] = VCN_PRTED_FENCE_WAIT_CMD;
	wptr_cpu[rb_len++] = 0; /* UVE__CMD_NATIVE_FENCE_TYPE__LNX_PROT */
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

	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			printf("\n%s: wait for 10ms done", __func__);
	}

#if FENCE_LECAGY
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in decode create test\n");
			break;
		}
		count++;
	}
#endif
	 r = amdgpu_bo_unmap_and_free_uq(device_handle, msg_buf.handle,
				     msg_buf.va_handle, msg_buf.addr,
				     PAGE_SIZE,
				     free_timeline_syncobj_handle, ++point5,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(free_timeline_syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle);
	drmSyncobjDestroy(device_handle->fd, free_timeline_syncobj_handle);
}

static void amdgpu_cs_vcn_dec_decode(void)
{
#if FENCE_LECAGY
	static int count;
	int test_pattern = 0xBADACAFE;
#endif
	const unsigned dpb_size = 15923584, dt_size = 737280;
	uint64_t msg_addr, fb_addr, bs_addr, dpb_addr, ctx_addr, dt_addr, it_addr, sum;
	struct amdgpu_vcn_bo dec_buf;
	uint64_t gtt_flags, point4, point6;
	uint32_t timeline_syncobj_handle4, timeline_syncobj_handle6;
	struct drm_amdgpu_userq_fence_info *fence_info;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	uint8_t *dec;
	uint32_t *ref_val_lo, *ref_val_hi;
	uint32_t syncobj_handle, syncarray[2];
	uint64_t s_handle;
	uint64_t gpu_addr, reference_val;
	int size, len, r, i;
	int syncobj_fd;

	fence_info = NULL;
	gtt_flags = point4 = point6 = 0;
	size = 4*1024; /* msg */
	size += 4*1024; /* fb */
	size += 4096; /*it_scaling_table*/
	size += ALIGN(sizeof(uvd_bitstream), 4*1024);
	size += ALIGN(dpb_size, 4*1024);
	size += ALIGN(dt_size, 4*1024);

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle4);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle6);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle,  size, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&dec_buf.handle, &dec_buf.ptr,
				&dec_buf.addr, &dec_buf.va_handle,
				timeline_syncobj_handle4, ++point4);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle4);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_map(dec_buf.handle, (void **)&dec_buf.ptr);
	dec = dec_buf.ptr;

	CU_ASSERT_EQUAL(r, 0);
	memset(dec_buf.ptr, 0, size);
	memcpy(dec_buf.ptr, vcn_dec_decode_msg, sizeof(vcn_dec_decode_msg));
	memcpy(dec_buf.ptr + sizeof(vcn_dec_decode_msg),
			avc_decode_msg, sizeof(avc_decode_msg));

	dec += 4*1024;
	memcpy(dec, feedback_msg, sizeof(feedback_msg));
	dec += 4*1024;
	memcpy(dec, uvd_it_scaling_table, sizeof(uvd_it_scaling_table));

	dec += 4*1024;
	memcpy(dec, uvd_bitstream, sizeof(uvd_bitstream));

	dec += ALIGN(sizeof(uvd_bitstream), 4*1024);

	dec += ALIGN(dpb_size, 4*1024);

	msg_addr = dec_buf.addr;
	fb_addr = msg_addr + 4*1024;
	it_addr = fb_addr + 4*1024;
	bs_addr = it_addr + 4*1024;
	dpb_addr = ALIGN(bs_addr + sizeof(uvd_bitstream), 4*1024);
	ctx_addr = ALIGN(dpb_addr + 0x006B9400, 4*1024);
	dt_addr = ALIGN(dpb_addr + dpb_size, 4*1024);

	memset(fence_cpu, 0, PAGE_SIZE);
	len = 0;
	vcn_dec_cmd(session_ctx_buf.addr, 0x5, &len);
	vcn_dec_cmd(msg_addr, 0x0, &len);
	vcn_dec_cmd(dpb_addr, 0x1, &len);
	vcn_dec_cmd(dt_addr, 0x2, &len);
	vcn_dec_cmd(fb_addr, 0x3, &len);
	vcn_dec_cmd(bs_addr, 0x100, &len);
	vcn_dec_cmd(it_addr, 0x204, &len);
	vcn_dec_cmd(ctx_addr, 0x206, &len);

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);


	/* add protected multifence fence */
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
	wptr_cpu[rb_len++] = test_pattern; // BADACAFE
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
	rb_len = *wptr_va_cpu;
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);

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
	wptr_cpu[rb_len++] = 0; // UVE__CMD_NATIVE_FENCE_TYPE__LNX_PROT
	wptr_cpu[rb_len++] = lower_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = upper_32_bits(gpu_addr);
	wptr_cpu[rb_len++] = lower_32_bits(reference_val);
	wptr_cpu[rb_len++] = upper_32_bits(reference_val);
	wptr_cpu[rb_len++] = VCN_ENC_CMD_END;

	*wptr_va_cpu = ALIGN(rb_len, 16);
	rb_len = *wptr_va_cpu;

#if LEGACY_FENCE
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in decode test\n");
			break;
		}
		count++;
	}
#endif
	for (i = 784; i < 787; i++)
		agdb_doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	for (i = 784; i < 787; i++)
		doorbell_ptr[VCN_DOORBELL_INDEX + i] = rb_len;

	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			printf("\n%s: wait for 10ms done", __func__);
	}

	for (i = 0, sum = 0; i < dt_size; ++i)
		sum += dec[i];

	CU_ASSERT_EQUAL(sum, SUM_DECODE);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, dec_buf.handle,
				     dec_buf.va_handle, dec_buf.addr,
				     size,
				     timeline_syncobj_handle6, ++point6,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle6);
	CU_ASSERT_EQUAL(r, 0);

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle4);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle6);
}

static void amdgpu_cs_vcn_dec_destroy(void)
{
	struct amdgpu_vcn_bo msg_buf;
	uint32_t timeline_syncobj_handle7, timeline_syncobj_handle8;
	uint64_t gtt_flags, point7, point8;
	struct drm_amdgpu_userq_signal signal_data;
	struct drm_amdgpu_userq_wait wait_data;
	struct drm_amdgpu_userq_fence_info *fence_info;
	uint32_t *ref_val_lo, *ref_val_hi;
	uint32_t syncobj_handle, syncarray[2];
	uint64_t s_handle;
	uint64_t gpu_addr, reference_val;
	int syncobj_fd, i;
	int len, r;
#if FENCE_LECAGY
	static int count;
	int test_pattern = 0xBAD0CAFE;
#endif

	fence_info = NULL;
	gtt_flags = point7 = point8 = 0;
	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle7);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjCreate(device_handle->fd, 0, &timeline_syncobj_handle8);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_alloc_and_map_uq(device_handle, PAGE_SIZE, PAGE_SIZE,
				AMDGPU_GEM_DOMAIN_GTT,
				gtt_flags,
				AMDGPU_VM_MTYPE_UC,
				&msg_buf.handle, &msg_buf.ptr,
				&msg_buf.addr, &msg_buf.va_handle,
				timeline_syncobj_handle7, ++point7);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle7);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_cpu_map(msg_buf.handle, (void **)&msg_buf.ptr);
	CU_ASSERT_EQUAL(r, 0);

	memset(msg_buf.ptr, 0, 1024);
	memcpy(msg_buf.ptr, vcn_dec_destroy_msg, sizeof(vcn_dec_destroy_msg));

	len = 0;
	vcn_dec_cmd(session_ctx_buf.addr, 5, &len);
	if (vcn_dec_sw_ring == true)
		vcn_dec_cmd(msg_buf.addr, 0, &len);

	if (vcn_unified_ring)
		amdgpu_cs_sq_ib_tail(ib_cpu + len);

	memset(fence_cpu, 0, PAGE_SIZE);

	/* add protected multifence */
	r = drmSyncobjCreate(device_handle->fd, 0, &syncobj_handle);
	CU_ASSERT_EQUAL(r, 0);

	r = drmSyncobjHandleToFD(device_handle->fd, syncobj_handle, &syncobj_fd);
	CU_ASSERT_EQUAL(r, 0);

	syncarray[0] = syncobj_handle;
	s_handle = syncobj_handle;

	rb_len++;
	wptr_cpu[rb_len++] = VCN_HDP_FLUSH_CMD;
	wptr_cpu[rb_len++] = VCN_ENC_CMD_IB;
	wptr_cpu[rb_len++] = 0;
	wptr_cpu[rb_len++] = lower_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = upper_32_bits(ib_ptr.addr);
	wptr_cpu[rb_len++] = len;

#if LEGACY_FENCE
	wptr_cpu[rb_len++] = VCN_ENC_CMD_FENCE;
	wptr_cpu[rb_len++] = lower_32_bits(fence.addr);
	wptr_cpu[rb_len++] = upper_32_bits(fence.addr);
	wptr_cpu[rb_len++] = test_pattern; // BAD0CAFE
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

	*wptr_va_cpu = ALIGN(rb_len, 16);
	rb_len = *wptr_va_cpu;
	*ref_val_lo = lower_32_bits(*wptr_va_cpu);
	*ref_val_hi = upper_32_bits(*wptr_va_cpu);

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

#if LEGACY_FENCE
	while (*fence_cpu != test_pattern) {
		usleep(10);
		if (count >= 10) {
			printf("\nTIMEOUT in decode destroy test\n");
			break;
		}
		count++;
	}

	CU_ASSERT_EQUAL(*fence_cpu, test_pattern);
#endif
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
	rb_len++;
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

	i = 0;
	while (*rptr_va_cpu_64 < reference_val) {
		usleep(10);
		i++;
		if (i > 1000)
			printf("\n%s: wait for 10ms done", __func__);
	}

	amdgpu_vcn_userqueue_destroy();

	r = amdgpu_bo_unmap_and_free_uq(device_handle, msg_buf.handle,
				     msg_buf.va_handle, msg_buf.addr,
				     PAGE_SIZE,
				     timeline_syncobj_handle8, ++point8,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = amdgpu_bo_unmap_and_free_uq(device_handle, session_ctx_buf.handle,
				     session_ctx_buf.va_handle, session_ctx_buf.addr,
				     32 * PAGE_SIZE,
				     timeline_syncobj_handle8, ++point8,
				     0, 0);
	CU_ASSERT_EQUAL(r, 0);

	r = timeline_syncobj_wait(timeline_syncobj_handle8);
	CU_ASSERT_EQUAL(r, 0);

	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle7);
	drmSyncobjDestroy(device_handle->fd, timeline_syncobj_handle8);
}
