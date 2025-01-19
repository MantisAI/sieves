from __future__ import annotations

import configparser
import importlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pydantic
import yaml  # type: ignore[import-untyped]

META_ATTRIBUTES = ("cls_name", "version")


class Attribute(pydantic.BaseModel):
    """Single attribute."""

    is_placeholder: bool
    value: Any

    @pydantic.model_validator(mode="after")
    def check_value(self) -> Attribute:
        """Validates .value property.
        :returns: Validated object.
        """
        # Adjust value to be class MODULE.NAME if is_placeholder.
        if self.is_placeholder and not isinstance(self.value, str):
            self.value = f"{self.value.__class__.__module__}.{self.value.__class__.__name__}"
        return self


class Config(pydantic.BaseModel):
    """Object representation."""

    @staticmethod
    def get_version() -> str:
        """Read version from setup.cfg.
        :return: Version string from setup.cfg metadata.
        """
        config = configparser.ConfigParser()
        setup_cfg = Path(__file__).parent.parent / "setup.cfg"
        config.read(setup_cfg)
        return config["metadata"]["version"]

    version: str = get_version()
    cls_name: str

    @classmethod
    def create(cls, cls_obj: type, attributes: dict[str, Attribute]) -> Config:
        """Creates config class, returns instance of this class.
        :param cls_obj: Fully qualified name of executing class.
        :param attributes: Attributes to inject.
        :return: Instane of dynamic config class.
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
        :returns: Config instance generate from dict representation. Config class.
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
        """Converts to fully qualified dict representation of init params for class.
        :param cls_obj: Type to be initiated with init dict.
        :param kwargs: Kwargs to inject for placeholder attributes.
        :returns: Fully qualified dict representation of init params for class.
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
        """Loads from .yml file.
        :param path: Pat to load from.
        :return: Config as stored at specified path.
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
        """Serializes to file.
        :returns: Representation instance.
        """

    @classmethod
    def deserialize(cls, config: Config, **kwargs: dict[str, Any]) -> Serializable:
        """Generate class instance from config.
        :param config: Config to generate instance from.
        :param kwargs: Values to inject into loaded config.
        :returns: Deserialized instance.
        """
