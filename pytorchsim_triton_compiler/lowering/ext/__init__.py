"""The target packs: one package per accelerator family (contract.target.ACCELERATORS), whose PROVIDES
names the module filling each pipeline Slot (passes.slots). A pack is the machine's definition: the
slots it fills are the units it has. A package here without PROVIDES is no pack but a store of generic
providers a pack may name (`<package>.<module>`)."""
