/*
 * Copyright 2024 Advanced Micro Devices, Inc.
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
#ifndef _VCN_TESTS__H_
#define _VCN_TESTS__H_

#define H264_NAL_TYPE_NON_IDR_SLICE 1
#define H264_NAL_TYPE_DP_A_SLICE 2
#define H264_NAL_TYPE_DP_B_SLICE 3
#define H264_NAL_TYPE_DP_C_SLICE 0x4
#define H264_NAL_TYPE_IDR_SLICE 0x5
#define H264_NAL_TYPE_SEI 0x6
#define H264_NAL_TYPE_SEQ_PARAM 0x7
#define H264_NAL_TYPE_PIC_PARAM 0x8
#define H264_NAL_TYPE_ACCESS_UNIT 0x9
#define H264_NAL_TYPE_END_OF_SEQ 0xa
#define H264_NAL_TYPE_END_OF_STREAM 0xb
#define H264_NAL_TYPE_FILLER_DATA 0xc
#define H264_NAL_TYPE_SEQ_EXTENSION 0xd

#define H264_START_CODE 0x000001

struct amdgpu_vcn_bo {
	amdgpu_bo_handle handle;
	amdgpu_va_handle va_handle;
	uint64_t addr;
	uint64_t size;
	void *ptr;
};

typedef struct BufferInfo_t {
	uint32_t numOfBitsInBuffer;
	const uint8_t *decBuffer;
	uint8_t decData;
	uint32_t decBufferSize;
	const uint8_t *end;
} bufferInfo;

typedef struct h264_decode_t {
	uint8_t profile;
	uint8_t level_idc;
	uint8_t nal_ref_idc;
	uint8_t nal_unit_type;
	uint32_t pic_width, pic_height;
	uint32_t slice_type;
} h264_decode;



inline int bs_eof(bufferInfo *bufinfo)
{
	if (bufinfo->decBuffer >= bufinfo->end)
		return 1;
	else
		return 0;
}

inline uint32_t bs_read_u1(bufferInfo *bufinfo)
{
	uint32_t r = 0;
	uint32_t temp = 0;

	bufinfo->numOfBitsInBuffer--;
	if (!bs_eof(bufinfo)) {
		temp = (((bufinfo->decData)) >> bufinfo->numOfBitsInBuffer);
		r = temp & 0x01;
	}

	if (bufinfo->numOfBitsInBuffer == 0) {
		bufinfo->decBuffer++;
		bufinfo->decData = *bufinfo->decBuffer;
		bufinfo->numOfBitsInBuffer = 8;
	}

	return r;
}

inline uint32_t bs_read_u(bufferInfo *bufinfo, int n)
{
	uint32_t r = 0;
	int i;

	for (i = 0; i < n; i++)
		r |= (bs_read_u1(bufinfo) << (n - i - 1));

	return r;
}

inline uint32_t bs_read_ue(bufferInfo *bufinfo)
{
	int32_t r = 0;
	int i = 0;

	while ((bs_read_u1(bufinfo) == 0) && (i < 32) && (!bs_eof(bufinfo)))
		i++;

	r = bs_read_u(bufinfo, i);
	r += (1 << i) - 1;
	return r;
}

void h264_check_0s(bufferInfo *bufinfo, int count);
int32_t h264_se(bufferInfo *bufinfo);
uint32_t remove_03(uint8_t *bptr, uint32_t len);
void scaling_list(uint32_t ix, uint32_t sizeOfScalingList, bufferInfo *bufInfo);
void h264_parse_sequence_parameter_set(h264_decode *dec, bufferInfo *bufInfo);
void h264_slice_header(h264_decode *dec, bufferInfo *bufInfo);
uint8_t h264_parse_nal(h264_decode *dec, bufferInfo *bufInfo);
uint32_t h264_find_next_start_code(uint8_t *pBuf, uint32_t bufLen);
#endif // _VCN_TESTS__H
