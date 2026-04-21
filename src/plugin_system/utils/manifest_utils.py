import re
from typing import Dict, Any, Tuple
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("清单工具")

MMC_VERSION = getattr(global_config, "MMC_VERSION", "1.0.0")


class VersionComparator:
    COMPATIBILITY_MAP: Dict[str, list] = {
        "0.8.0": [
            "0.8.1",
            "0.8.2",
            "0.8.3",
            "0.8.4",
            "0.8.5",
            "0.8.6",
            "0.8.7",
            "0.8.8",
            "0.8.9",
            "0.8.10",
        ],
        "0.8.1": [
            "0.8.2",
            "0.8.3",
            "0.8.4",
            "0.8.5",
            "0.8.6",
            "0.8.7",
            "0.8.8",
            "0.8.9",
            "0.8.10",
        ],
        "0.8.2": [
            "0.8.3",
            "0.8.4",
            "0.8.5",
            "0.8.6",
            "0.8.7",
            "0.8.8",
            "0.8.9",
            "0.8.10",
        ],
        "0.8.3": [
            "0.8.4",
            "0.8.5",
            "0.8.6",
            "0.8.7",
            "0.8.8",
            "0.8.9",
            "0.8.10",
        ],
        "0.8.4": ["0.8.5", "0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10"],
        "0.8.5": ["0.8.6", "0.8.7", "0.8.8", "0.8.9", "0.8.10"],
        "0.8.6": ["0.8.7", "0.8.8", "0.8.9", "0.8.10"],
        "0.8.7": ["0.8.8", "0.8.9", "0.8.10"],
        "0.8.8": ["0.8.9", "0.8.10"],
        "0.8.9": ["0.8.10"],
    }

    @staticmethod
    def normalize_version(version: str) -> str:
        if not version:
            return "0.0.0"
        version = version.strip()
        match = re.match(
            r"^v?(\d+(?:\.\d+){0,2})(?:[-+][0-9A-Za-z.-]+)?$", version
        )
        if not match:
            return "0.0.0"
        normalized = match.group(1)
        parts = normalized.split(".")
        while len(parts) < 3:
            parts.append("0")
        normalized = ".".join(parts[:3])
        return normalized

    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int]:
        normalized = VersionComparator.normalize_version(version)
        try:
            parts = normalized.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            logger.warning(f"无法解析版本号: {version}，使用默认版本 0.0.0")
            return (0, 0, 0)

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        v1_tuple = VersionComparator.parse_version(version1)
        v2_tuple = VersionComparator.parse_version(version2)
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0

    @staticmethod
    def check_forward_compatibility(
        current_version: str, max_version: str
    ) -> Tuple[bool, str]:
        current_normalized = VersionComparator.normalize_version(
            current_version
        )
        max_normalized = VersionComparator.normalize_version(max_version)
        if max_normalized in VersionComparator.COMPATIBILITY_MAP:
            compatible_versions = VersionComparator.COMPATIBILITY_MAP[
                max_normalized
            ]
            if current_normalized in compatible_versions:
                return (
                    True,
                    f"根据兼容性映射表，版本 {current_normalized} 与 {max_normalized} 兼容",
                )
        return False, ""

    @staticmethod
    def is_version_in_range(
        version: str, min_version: str = "", max_version: str = ""
    ) -> Tuple[bool, str]:
        if not min_version and not max_version:
            return True, ""
        version_normalized = VersionComparator.normalize_version(version)
        if min_version:
            min_normalized = VersionComparator.normalize_version(min_version)
            if (
                VersionComparator.compare_versions(
                    version_normalized, min_normalized
                )
                < 0
            ):
                return (
                    False,
                    f"版本 {version_normalized} 低于最小要求版本 {min_normalized}",
                )
        if max_version:
            max_normalized = VersionComparator.normalize_version(max_version)
            comparison = VersionComparator.compare_versions(
                version_normalized, max_normalized
            )
            if comparison > 0:
                is_compatible, compat_msg = (
                    VersionComparator.check_forward_compatibility(
                        version_normalized, max_normalized
                    )
                )
                if not is_compatible:
                    return (
                        False,
                        f"版本 {version_normalized} 高于最大支持版本 {max_normalized}，且无兼容性映射",
                    )
                logger.info(f"版本兼容性检查：{compat_msg}")
                return True, compat_msg
        return True, ""

    @staticmethod
    def get_current_host_version() -> str:
        return VersionComparator.normalize_version(MMC_VERSION)

    @staticmethod
    def add_compatibility_mapping(
        base_version: str, compatible_versions: list
    ) -> None:
        base_normalized = VersionComparator.normalize_version(base_version)
        VersionComparator.COMPATIBILITY_MAP[base_normalized] = [
            VersionComparator.normalize_version(v) for v in compatible_versions
        ]
        logger.info(
            f"添加兼容性映射：{base_normalized} -> {compatible_versions}"
        )

    @staticmethod
    def get_compatibility_info() -> Dict[str, list]:
        return VersionComparator.COMPATIBILITY_MAP.copy()


class ManifestValidator:
    REQUIRED_FIELDS = [
        "manifest_version",
        "name",
        "version",
        "description",
        "author",
    ]
    OPTIONAL_FIELDS = [
        "license",
        "host_application",
        "homepage_url",
        "repository_url",
        "keywords",
        "categories",
        "default_locale",
        "locales_path",
        "plugin_info",
    ]
    RECOMMENDED_FIELDS = ["license", "keywords", "categories"]
    SUPPORTED_MANIFEST_VERSIONS = [1]

    def __init__(self):
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []

    def validate_manifest(self, manifest_data: Dict[str, Any]) -> bool:
        self.validation_errors.clear()
        self.validation_warnings.clear()
        for field in self.REQUIRED_FIELDS:
            if field not in manifest_data:
                self.validation_errors.append(f"缺少必需字段: {field}")
            elif not manifest_data[field]:
                self.validation_errors.append(f"必需字段不能为空: {field}")
        if "manifest_version" in manifest_data:
            version = manifest_data["manifest_version"]
            if version not in self.SUPPORTED_MANIFEST_VERSIONS:
                self.validation_errors.append(
                    f"不支持的manifest版本: {version}，支持的版本: {
                        self.SUPPORTED_MANIFEST_VERSIONS}"
                )
        if "author" in manifest_data:
            author = manifest_data["author"]
            if isinstance(author, dict):
                if "name" not in author or not author["name"]:
                    self.validation_errors.append("作者信息缺少name字段或为空")
                if "url" in author and author["url"]:
                    url = author["url"]
                    if not (
                        url.startswith("http://") or url.startswith("https://")
                    ):
                        self.validation_warnings.append(
                            "作者URL建议使用完整的URL格式"
                        )
            elif isinstance(author, str):
                if not author.strip():
                    self.validation_errors.append("作者信息不能为空")
            else:
                self.validation_errors.append(
                    "作者信息格式错误，应为字符串或包含name字段的对象"
                )
        if "host_application" in manifest_data:
            host_app = manifest_data["host_application"]
            if isinstance(host_app, dict):
                min_version = host_app.get("min_version", "")
                max_version = host_app.get("max_version", "")
                for version_field in ["min_version", "max_version"]:
                    if (
                        version_field in host_app
                        and not host_app[version_field]
                    ):
                        self.validation_warnings.append(
                            f"host_application.{version_field}为空"
                        )
                if min_version or max_version:
                    current_version = (
                        VersionComparator.get_current_host_version()
                    )
                    is_compatible, error_msg = (
                        VersionComparator.is_version_in_range(
                            current_version, min_version, max_version
                        )
                    )
                    if not is_compatible:
                        self.validation_errors.append(
                            f"版本兼容性检查失败: {error_msg} (当前版本: {current_version})"
                        )
                    else:
                        logger.debug(
                            f"版本兼容性检查通过: 当前版本 {current_version} 符合要求 [{min_version}, {max_version}]"
                        )
            else:
                self.validation_errors.append(
                    "host_application格式错误，应为对象"
                )
        for url_field in ["homepage_url", "repository_url"]:
            if url_field in manifest_data and manifest_data[url_field]:
                url: str = manifest_data[url_field]
                if not (
                    url.startswith("http://") or url.startswith("https://")
                ):
                    self.validation_warnings.append(
                        f"{url_field}建议使用完整的URL格式"
                    )
        for list_field in ["keywords", "categories"]:
            if list_field in manifest_data:
                field_value = manifest_data[list_field]
                if field_value is not None and not isinstance(
                    field_value, list
                ):
                    self.validation_errors.append(f"{list_field}应为数组格式")
                elif isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        if not isinstance(item, str):
                            self.validation_warnings.append(
                                f"{list_field}[{i}]应为字符串"
                            )
        for field in self.RECOMMENDED_FIELDS:
            if field not in manifest_data or not manifest_data[field]:
                self.validation_warnings.append(f"建议填写字段: {field}")
        if "plugin_info" in manifest_data:
            plugin_info = manifest_data["plugin_info"]
            if isinstance(plugin_info, dict):
                if "components" in plugin_info:
                    components = plugin_info["components"]
                    if not isinstance(components, list):
                        self.validation_errors.append(
                            "plugin_info.components应为数组格式"
                        )
                    else:
                        for i, component in enumerate(components):
                            if not isinstance(component, dict):
                                self.validation_errors.append(
                                    f"plugin_info.components[{i}]应为对象"
                                )
                            else:
                                for comp_field in [
                                    "type",
                                    "name",
                                    "description",
                                ]:
                                    if (
                                        comp_field not in component
                                        or not component[comp_field]
                                    ):
                                        self.validation_errors.append(
                                            f"plugin_info.components[{i}]缺少必需字段: {comp_field}"
                                        )
            else:
                self.validation_errors.append("plugin_info应为对象格式")
        return len(self.validation_errors) == 0

    def get_validation_report(self) -> str:
        report = []
        if self.validation_errors:
            report.append("验证错误:")
            report.extend(f"  - {error}" for error in self.validation_errors)
        if self.validation_warnings:
            report.append("验证警告:")
            report.extend(
                f"  - {warning}" for warning in self.validation_warnings
            )
        if not self.validation_errors and not self.validation_warnings:
            report.append("Manifest文件验证通过")
        return "\n".join(report)

    def get_errors(self) -> list[str]:
        return self.validation_errors.copy()

    def get_warnings(self) -> list[str]:
        return self.validation_warnings.copy()


def validate_manifest_data(
    manifest_data: Dict[str, Any],
) -> Tuple[bool, list[str], list[str]]:
    validator = ManifestValidator()
    is_valid = validator.validate_manifest(manifest_data)
    return is_valid, validator.get_errors(), validator.get_warnings()


def check_version_compatibility(
    min_version: str = "", max_version: str = ""
) -> Tuple[bool, str]:
    current_version = VersionComparator.get_current_host_version()
    return VersionComparator.is_version_in_range(
        current_version, min_version, max_version
    )
