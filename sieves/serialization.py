from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pydantic
import yaml  # type: ignore[import-untyped]

META_ATTRIBUTES = ("cls_name", "version")


class Attribute(pydantic.BaseModel):
    """Single attribute."""

    value: Any
    is_placeholder: bool | None = None

    @staticmethod
    def _is_primitive_type(value: Any) -> bool:
        """Determines whether value is primitive type.
        :param value: Value to check.
        :return bool: Whether value is primitive type.
        """
        return any([isinstance(value, t) for t in (set, int, float, str, Config)])

    @classmethod
    def _determine_is_placeholder(cls, value: Any) -> bool:
        """Determines whether Attribute value is a non-supported complex type and hence a placeholder.
        If value is a collection type, we inspect recursively. If all children elements are primitive types, values is
        determined not be a placeholder.
        :return bool: Determined value for is_placeholder.
        """

        # If value is None or a primitive type or a Config object: not a placeholder.
        if value is None or cls._is_primitive_type(value):
            return False

        # Investigate collection types.
        if isinstance(value, dict):
            return any(cls._determine_is_placeholder(child) for child in value.values())
        elif any([isinstance(value, t) for t in (list, tuple, set)]):
            return any(cls._determine_is_placeholder(child) for child in value)
        # todo Support serialization of Pydantic classes (convert to dict to determine placeholder status).

        # Unknown value type: assume placeholder status.
        return True

    @pydantic.model_validator(mode="after")
    def check_value(self) -> Attribute:
        """Validates value and returns validated object.
        :return Attribute: Validated object.
        """
        # Set is_placeholder w.r.t. of value type.
        if self.is_placeholder is None:
            self.is_placeholder = Attribute._determine_is_placeholder(self.value)

        # Adjust value to be class MODULE.NAME if is_placeholder.
        if self.is_placeholder and not isinstance(self.value, str):
            if self.value is not None:
                if hasattr(self.value, "__class__"):
                    self.value = f"{self.value.__class__.__module__}.{self.value.__class__.__name__}"
                else:
                    self.value = getattr(self.value, "__name__", "Unknown")

        return self


class Config(pydantic.BaseModel):
    """Object representation."""

    @staticmethod
    def get_version() -> str:
        """Returns version string from setup.cfg metadata.
        :return str: Version string from setup.cfg metadata.
        """
        # TODO This doesn't work in PyPi package. Alternative way to fetch version dynamically?
        # config = configparser.ConfigParser()
        # setup_cfg = Path(__file__).parent.parent / "setup.cfg"
        # config.read(setup_cfg)
        # version = config["metadata"]["version"]

        return "0.11.1"

    version: str = get_version()
    cls_name: str

    @classmethod
    def create(cls, cls_obj: type, attributes: dict[str, Attribute]) -> Config:
        """Creates instance of dynamic config class.
        :param cls_obj: Class to create config for.
        :param attributes: Attributes to include in config.
        :return Config: Instance of dynamic config class.
        """
        config_type = pydantic.create_model(  # type: ignore[call-overload]
            f"{cls_obj}Config",
            __base__=Config,
            **{attr_id: (Attribute, ...) for attr_id in attributes},
        )
        config = config_type(cls_name=f"{cls_obj.__module__}.{cls_obj.__name__}", **attributes)
        assert isinstance(config, Config)

        return config

    @property
    def config_cls(self) -> type:
        module_name, class_name = self.cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        config_cls = getattr(module, class_name)
        assert isinstance(config_cls, type)

        return config_cls

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> tuple[Config, type]:
        """Generates Config instance from dict representation.
        :param config: Dict representation of config.
        :return tuple[Config, type]: Config instance generate from dict representation. Config class.
        """
        # Dynamically import class.
        module_name, class_name = config["cls_name"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        config_cls = getattr(module, class_name)

        # Convert value information to Attribute instances.
        attributes = {
            attr_name: Attribute(**attr)
            for attr_name, attr in config.items()
            # Ignore meta attributes.
            if attr_name not in META_ATTRIBUTES
        }

        return Config.create(config_cls, attributes), config_cls

    def to_init_dict(self, cls_obj: type, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """Returns fully qualified dict representation of init params for class.
        :param cls_obj: Class to get init params for.
        :param kwargs: Values to inject into loaded config.
        :return dict[str, Any]: Fully qualified dict representation of init params for class.
        """
        self.validate_init_params(cls_obj, **kwargs)
        config_params = self.model_dump()

        init_params = {
            p_name: p_value["value"] for p_name, p_value in config_params.items() if p_name not in META_ATTRIBUTES
        } | kwargs

        return init_params

    def validate_init_params(self, cls_obj: type, **kwargs: dict[str, Any]) -> None:
        """Validates config against class type and kwargs to inject.
        :param cls_obj: Type of object to instantiate.
        :param kwargs: kwargs to inject into init dict (to replace placeholder values).
        """
        # Assert metadata is correct.
        assert self.version == Config.get_version()
        assert self.cls_name == f"{cls_obj.__module__}.{cls_obj.__name__}"

        # Assert all placeholder attributes are provided.
        for attr_name, attr in self.model_dump().items():
            # Skip meta attributes.
            if attr_name in META_ATTRIBUTES:
                continue
            if attr["is_placeholder"]:
                assert attr_name in kwargs, f"Attribute {attr_name} has to be provided at load time."

    def dump(self, path: Path | str) -> None:
        """Saves to disk as .yml file.
        :param path: Path to save at.
        """
        with open(path, "w") as file:
            yaml.dump(self.model_dump(mode="json"), file)

    @classmethod
    def load(cls, path: Path | str) -> Config:
        """Loads config from specified yml path.
        :param path: Path to load config from.
        :return Config: Config as stored at specified path.
        """
        with open(path) as file:
            data = yaml.safe_load(file)

        config = pydantic.create_model(  # type: ignore[call-overload]
            f"{data['cls_name']}Config",
            __base__=Config,
            cls_name=(str, ...),
            **{attr_id: (Attribute, ...) for attr_id in data if attr_id not in META_ATTRIBUTES},
        )

        loaded_config = config.model_validate(data)
        assert isinstance(loaded_config, Config)

        return loaded_config


@runtime_checkable
class Serializable(Protocol):
    def serialize(self) -> Config:
        """Returns serialized representation.
        :return Config: Representation instance.
        """

    @classmethod
    def deserialize(cls, config: Config, **kwargs: dict[str, Any]) -> Serializable:
        """Returns deserialized instance.
        :param config: Config to deserialize.
        :param kwargs: Values to inject into loaded config.
        :return Serializable: Deserialized instance.
        """
