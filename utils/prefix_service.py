# prefix_service.py
import re
from dataclasses import dataclass
from typing import Optional

try:
    from infrastructure.logger_config import setup_logger
    from core.exceptions import BtgTranslationException
except ImportError:
    from infrastructure.logger_config import setup_logger  # type: ignore
    from core.exceptions import BtgTranslationException  # type: ignore

logger = setup_logger(__name__)

# [00001]| 형식 매칭 (줄 시작, 5자리 0패딩 숫자, 대괄호, 파이프)
_PREFIX_PATTERN = re.compile(r'^\[(\d{5})\]\|(.*)', re.DOTALL)
_PREFIX_FORMAT = '[{:05d}]|'

# LLM 조기 종료를 유발하는 노이즈 패턴 — 빈 줄로 취급 (번호 미부착, 원문 그대로 복원)
_NOISE_PATTERNS = re.compile(
    r'^\s*[（(]'           # 여는 괄호 (전각/반각)
    r'(本章完|完結|完|終|End|THE END|fin)'  # 종결 표시
    r'[）)]\s*$',          # 닫는 괄호
    re.IGNORECASE
)


@dataclass
class LineMetadata:
    """청크 내 각 줄의 메타데이터"""
    prefix_num: Optional[int]   # 비어있는 줄은 None
    global_line_number: int     # 원본 파일 기준 1-based 줄 번호
    original_text: str          # 줄 끝 개행 제외한 원문 내용
    is_empty: bool              # 빈 줄(공백만 있는 줄 포함) 여부


class PrefixService:
    """번역 완전성 검증을 위한 프리픽스 기반 서비스.

    각 비어있지 않은 줄에 [NNNNN] 형식의 번호를 붙여 번역하고,
    번역 후 번호 대조를 통해 누락된 문장을 원문으로 복원합니다.
    """

    def add_prefixes_to_chunk(
        self,
        chunk_text: str,
        global_line_offset: int = 0
    ) -> tuple[str, list[LineMetadata]]:
        """청크의 각 비어있지 않은 줄 앞에 [NNNNN] 프리픽스를 추가합니다.

        빈 줄(공백만 있는 줄 포함)은 프리픽스를 붙이지 않고 그대로 유지합니다.
        프리픽스 번호는 청크 내에서 1부터 시작하며, 다음 청크에서 리셋됩니다.

        Args:
            chunk_text: 원본 청크 텍스트
            global_line_offset: 이 청크 첫 번째 줄의 원본 파일 내 0-based 줄 인덱스
                                 (예: 이전 청크까지의 총 줄 수)

        Returns:
            (프리픽스가 추가된 텍스트, 줄별 LineMetadata 리스트)
        """
        lines = chunk_text.splitlines(keepends=True)
        metadata_list: list[LineMetadata] = []
        prefixed_lines: list[str] = []
        prefix_counter = 0

        for i, line in enumerate(lines):
            global_line_number = global_line_offset + i + 1  # 1-based
            # 줄 끝 개행 문자 분리
            line_content = line.rstrip('\r\n')
            newline_suffix = line[len(line_content):]  # '\n', '\r\n', '' 등

            if not line_content.strip() or _NOISE_PATTERNS.match(line_content):
                # 빈 줄 or 노이즈 패턴(本章完 등): LLM에 빈 줄로 전송 (내용 숨김)
                if line_content.strip():
                    logger.debug(f"[프리픽스] 노이즈 패턴 감지, LLM에 빈 줄로 전송: {line_content!r}")
                metadata_list.append(LineMetadata(
                    prefix_num=None,
                    global_line_number=global_line_number,
                    original_text="" if line_content.strip() else line_content,
                    is_empty=True,
                ))
                prefixed_lines.append(newline_suffix or '\n')
            else:
                prefix_counter += 1
                metadata_list.append(LineMetadata(
                    prefix_num=prefix_counter,
                    global_line_number=global_line_number,
                    original_text=line_content,
                    is_empty=False,
                ))
                prefixed_lines.append(
                    f"{_PREFIX_FORMAT.format(prefix_counter)}{line_content}{newline_suffix}"
                )

        prefixed_text = ''.join(prefixed_lines)
        logger.info(
            f"[프리픽스] 부착 완료: {prefix_counter}개 비어있지 않은 줄 / 전체 {len(lines)}줄 "
            f"(global_offset={global_line_offset}, 번호 [00001]|~[{prefix_counter:05d}]|)"
        )
        return prefixed_text, metadata_list

    def parse_prefixed_translation(self, translated_text: str) -> dict[int, str]:
        """번역 결과에서 [NNNNN] 프리픽스를 파싱하여 {번호: 번역텍스트} 반환.

        같은 번호가 중복 등장하면 첫 번째 값을 유지합니다.
        매칭 결과가 0개이면 BtgTranslationException을 발생시켜 번역을 중단합니다.

        Args:
            translated_text: LLM이 출력한 번역 결과 전체 텍스트

        Returns:
            {prefix_num: translated_line} 딕셔너리

        Raises:
            BtgTranslationException: 프리픽스가 하나도 감지되지 않은 경우
        """
        result: dict[int, str] = {}

        for line in translated_text.splitlines():
            m = _PREFIX_PATTERN.match(line.strip())
            if m:
                num = int(m.group(1))
                text = m.group(2).strip()
                if num not in result:
                    result[num] = text
                else:
                    logger.warning(f"중복 프리픽스 [{num:05d}] 감지 - 첫 번째 값 유지")

        if not result:
            preview = translated_text[:300].replace('\n', '↵')
            logger.error(
                f"번역 결과에서 프리픽스를 하나도 찾을 수 없습니다. "
                f"번역 미리보기: {preview}"
            )
            raise BtgTranslationException(
                "번역 결과에 [NNNNN] 형식의 프리픽스가 없습니다. "
                "LLM이 프리픽스를 모두 제거한 것으로 보입니다. "
                "API 비용 낭비를 방지하기 위해 번역을 중단합니다."
            )

        logger.info(f"[프리픽스] 파싱 완료: {len(result)}개 항목 매칭")
        return result

    def reconstruct_output(
        self,
        line_metadata: list[LineMetadata],
        translated_map: dict[int, str],
        original_text_on_missing: bool = False,
    ) -> str:
        """원본 메타데이터와 번역 결과를 대조하여 최종 출력 텍스트 생성.

        처리 규칙:
        - 빈 줄 → 빈 줄 그대로 유지
        - 번역 있는 줄 → 번역 텍스트 (프리픽스 제거)
        - 번역 누락 줄 → original_text_on_missing=False이면 @offset:: 주석,
                         True이면 원문 텍스트 그대로

        Args:
            line_metadata: add_prefixes_to_chunk()가 반환한 메타데이터 리스트
            translated_map: parse_prefixed_translation()이 반환한 {번호: 번역} 딕셔너리
            original_text_on_missing: True이면 누락 줄에 원문을 출력 (재번역 후 fallback용)

        Returns:
            최종 출력 텍스트 (줄바꿈으로 연결)
        """
        output_lines: list[str] = []
        missing_count = 0

        for meta in line_metadata:
            if meta.is_empty:
                output_lines.append(meta.original_text)
            elif meta.prefix_num in translated_map:
                output_lines.append(translated_map[meta.prefix_num])
            else:
                missing_count += 1
                if original_text_on_missing:
                    output_lines.append(meta.original_text)
                else:
                    output_lines.append(
                        f'@offset::{meta.global_line_number} translate missing, '
                        f'original text "{meta.original_text}"'
                    )

        if missing_count > 0:
            if original_text_on_missing:
                logger.warning(f"번역 누락 {missing_count}개 줄 → 원문으로 대체")
            else:
                logger.warning(f"번역 누락 {missing_count}개 줄 → @offset 주석 삽입")

        return '\n'.join(output_lines)

    def count_chunk_lines(self, chunk_text: str) -> int:
        """청크의 총 줄 수를 반환합니다 (다음 청크의 global_line_offset 계산용)."""
        return len(chunk_text.splitlines())
