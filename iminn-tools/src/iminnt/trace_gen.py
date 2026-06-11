
import json
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Union
import itertools
from pathlib import Path
import re
import bisect

from elftools.elf.elffile import ELFFile
from elftools.common.utils import bytes2str
from elftools.dwarf.descriptions import describe_form_class

PC_INT_EXPR=re.compile(r"pc(\+|\-)(\d+)")
PC_HEX_EXPR=re.compile(r"pc(\+|\-)(0x[0-9a-fA-F]+)")
SKIP_FUNCS = ["__vfprintf_internal", "__libc_malloc", "nosym", "exit"]
NOSYM_NAME = "nosym"

def is_int(v: str):
    try:
        int(v)
    except ValueError:
        return False
    return True

def is_hex(v: str):
    try:
        int(v, 16)
    except ValueError:
        return False
    return True

REG_NAME_MAP = {
    "s0": "fp",
    "zero": "x0",
    "ra": "x1",
    "sp": "x2",
    "gp": "x3",
    "tp": "x4",
    "t0": "x5",
    "t1": "x6",
    "t2": "x7",
    "s0": "x8",
    "s1": "x9",
    "a0": "x10",
    "a1": "x11",
    "a2": "x12",
    "a3": "x13",
    "a4": "x14",
    "a5": "x15",
    "a6": "x16",
    "a7": "x17",
    "s2": "x18",
    "s3": "x19",
    "s4": "x20",
    "s5": "x21",
    "s6": "x22",
    "s7": "x23",
    "s8": "x24",
    "s9": "x25",
    "s10": "x26",
    "s11": "x27",
    "t3": "x28",
    "t4": "x29",
    "t5": "x30",
    "t6": "x31",
    "ft0": "f0",
    "ft1": "f1",
    "ft2": "f2",
    "ft3": "f3",
    "ft4": "f4",
    "ft5": "f5",
    "ft6": "f6",
    "ft7": "f7",
    "fs0": "f8",
    "fs1": "f9",
    "fa0": "f10",
    "fa1": "f11",
    "fa2": "f12",
    "fa3": "f13",
    "fa4": "f14",
    "fa5": "f15",
    "fa6": "f16",
    "fa7": "f17",
    "fs2": "f18",
    "fs3": "f19",
    "fs4": "f20",
    "fs5": "f21",
    "fs6": "f22",
    "fs7": "f23",
    "fs8": "f24",
    "fs9": "f25",
    "fs10": "f26",
    "fs11": "f27",
    "ft8": "f28",
    "ft9": "f29",
    "ft10": "f30",
    "ft11": "f31",
}

def get_instr_bin(f, elf, addr: int, size: int = 4):
    for sec in elf.iter_sections():
        start = sec['sh_addr']
        end   = start + sec['sh_size']
        if start <= addr < end:
            # virtual→file offset
            file_off = addr - start + sec['sh_offset']
            f.seek(file_off)
            return f.read(size).hex()

    raise ValueError(f"Address 0x{vaddr:x} not in any loaded section")

def decode_funcname(dwarfinfo, address):
    # Go over all DIEs in the DWARF information, looking for a subprogram
    # entry with an address range that includes the given address. Note that
    # this simplifies things by disregarding subprograms that may have
    # split address ranges.
    for CU in dwarfinfo.iter_CUs():
        for DIE in CU.iter_DIEs():
            try:
                if DIE.tag == 'DW_TAG_subprogram':
                    lowpc = DIE.attributes['DW_AT_low_pc'].value

                    # DWARF v4 in section 2.17 describes how to interpret the
                    # DW_AT_high_pc attribute based on the class of its form.
                    # For class 'address' it's taken as an absolute address
                    # (similarly to DW_AT_low_pc); for class 'constant', it's
                    # an offset from DW_AT_low_pc.
                    highpc_attr = DIE.attributes['DW_AT_high_pc']
                    highpc_attr_class = describe_form_class(highpc_attr.form)
                    if highpc_attr_class == 'address':
                        highpc = highpc_attr.value
                    elif highpc_attr_class == 'constant':
                        highpc = lowpc + highpc_attr.value
                    else:
                        print('Error: invalid DW_AT_high_pc class:',
                              highpc_attr_class)
                        continue

                    if lowpc <= address < highpc:
                        return bytes2str(DIE.attributes['DW_AT_name'].value)
            except KeyError:
                continue
    return "unknown"


def lpe_filename(line_program, file_index):
    # Retrieving the filename associated with a line program entry
    # involves two levels of indirection: we take the file index from
    # the LPE to grab the file_entry from the line program header,
    # then take the directory index from the file_entry to grab the
    # directory name from the line program header. Finally, we
    # join the (base) filename from the file_entry to the directory
    # name to get the absolute filename.
    lp_header = line_program.header
    file_entries = lp_header["file_entry"]

    # File and directory indices are 1-indexed in DWARF version < 5,
    # 0-indexed in DWARF5.
    if lp_header.version < 5:
        file_index -= 1
    if file_index == -1:
        return None

    file_entry = file_entries[file_index]
    dir_index = file_entry["dir_index"]

    # A dir_index of 0 indicates that no absolute directory was recorded during
    # compilation in DWARF version < 5; return just the basename.
    if dir_index == 0 and lp_header.version < 5:
        return file_entry.name.decode()

    if lp_header.version < 5:
        dir_index -= 1
    directory = lp_header["include_directory"][dir_index]
    return Path(directory.decode()) / file_entry.name.decode()

def decode_file_line(dwarfinfo, address):
    # Go over all the line programs in the DWARF information, looking for
    # one that describes the given address.
    for CU in dwarfinfo.iter_CUs():
        # First, look at line programs to find the file/line for the address
        lineprog = dwarfinfo.line_program_for_CU(CU)
        delta = 1 if lineprog.header.version < 5 else 0
        prevstate = None
        for entry in lineprog.get_entries():
            # We're interested in those entries where a new state is assigned
            if entry.state is None:
                continue
            # Looking for a range of addresses in two consecutive states that
            # contain the required address.
            if prevstate and prevstate.address <= address < entry.state.address:
                fpath = lpe_filename(lineprog, prevstate.file - delta)
                line = prevstate.line
                return fpath, line
            if entry.state.end_sequence:
                # For the state with `end_sequence`, `address` means the address
                # of the first byte after the target machine instruction
                # sequence and other information is meaningless. We clear
                # prevstate so that it's not used in the next iteration. Address
                # info is used in the above comparison to see if we need to use
                # the line information for the prevstate.
                prevstate = None
            else:
                prevstate = entry.state
    return None, None

@dataclass
class SymResolver:
    syms: List[Tuple[str, int, int]]
    fn_starts: List[int]
    elf_path: Path
    debug_info: bool

    def lookup_sym(self, addr: int) -> str:
        # Binary search for function
        idx = bisect.bisect_right(self.fn_starts, addr) - 1
        if idx >= 0 and idx < len(self.syms):
            fn, start, end = self.syms[idx]
            if addr >= start and addr < end:
                return fn
        return NOSYM_NAME

    @classmethod
    def from_file(cls, fpath: Path):
        syms = []

        with open(str(fpath), 'rb') as f:
            elffile = ELFFile(f)
            symtab = elffile.get_section_by_name('.symtab')
            # dwarf_info = elf.get_dwarf_info()
            # assert debug info
            for sym in symtab.iter_symbols():
                if sym['st_info']['type'] == 'STT_FUNC' and sym['st_size'] > 0:
                    start = sym['st_value']
                    end   = start + sym['st_size']
                    syms.append((sym.name, start, end))
                elif sym['st_info']['type'] == 'STT_NOTYPE' and sym.name:
                    print(f"{sym.name}")
        syms.sort(key=lambda f: f[1])
        fn_starts = [f[1] for f in syms]
        return cls(syms, fn_starts, fpath, elffile.has_dwarf_info())

    def add_debug_info(self, events: List[Dict]):
        debug_events = []
        with open(str(self.elf_path), 'rb') as f:
            elffile = ELFFile(f)
            dwarfinfo = elffile.get_dwarf_info()
            for e in events:
                db_event = e.copy()
                fn_addr = e["args"]["fn_addr"]
                caller_addr = e["args"]["caller_addr"]
                fn_fn = decode_funcname(dwarfinfo, fn_addr)
                fn_file, fn_line = decode_file_line(dwarfinfo, fn_addr)
                if fn_file is not None:
                    db_event["args"]["fn_info"] = f"{fn_file}:{fn_line}:{fn_fn}"
                call_fn = decode_funcname(dwarfinfo, caller_addr)
                call_file, call_line = decode_file_line(dwarfinfo, caller_addr)
                if call_file is not None:
                    db_event["args"]["caller_info"] = f"{call_file}:{call_line}:{call_fn}"
                debug_events.append(db_event)
        return debug_events

@dataclass
class Instruction:
    """Function information from ELF/DWARF"""
    mnem: str
    operands: List[Union[str, int]]
    cycle: int
    pc: int
    tid: int = 0
    pid: int = 0
    cat: str = "instr"
    fn: str = "none"

    def __str__(self):
        if len(self.operands) == 0:
            return self.mnem
        op_str = ",".join([o if isinstance(o, str) else str(hex(o)) for o in self.operands])
        return f"{self.mnem} {op_str}"

    @classmethod
    def none_instr(cls):
        return cls("none", [], -1, -1)

    @classmethod
    def from_str(cls, line: str):
        parts = line.strip().split("|")
        if len(parts) >= 3:
            try:
                cycle = int(parts[0].split(":")[1])
                pc = int(parts[3].split(":")[1], 16)
            except ValueError:
                return None
        instr = parts[-1].split(":")[1].split()
        mnem = instr[0].strip()
        if len(instr) == 1:
            return cls(mnem, [], cycle, pc, 0, 0)
        args = "".join(instr[1:]).split(",")
        operands = []
        for a_str in args:
            a = a_str.strip()
            if PC_INT_EXPR.search(a) or PC_HEX_EXPR.search(a):
                v = eval(a, {"pc": pc})
                operands.append(v)
            elif is_int(a):
                operands.append(int(a))
            elif is_hex(a):
                operands.append(int(a, 16))
            else:
                operands.append(REG_NAME_MAP.get(a, a))

        # Canonicalize expansions
        if len(operands) == 1 and mnem in ["j", "jal", "jr", "jalr"]:
            if (mnem == "j" and isinstance(operands[0], int)) or (mnem == "jr" and isinstance(operands[0], str)):
                operands.insert(0, "x0")
                mnem = "jal" if mnem == "j" else "jalr"

            if (mnem == "jal" and isinstance(operands[0], int)) or (mnem == "jalr" and isinstance(operands[0], str)):
                operands.insert(0, "x1")
        return cls(mnem.replace("\t", "").replace(" ", ""), operands, cycle, pc, 0, 0)

    # https://github.com/riscv-non-isa/riscv-trace-spec/blob/main/referenceFlow/post-iss/src/post_inst_set_sim.c#L438
    def is_call(self) -> bool:
        # jal x1 or jal x5
        if self.mnem == "jal" and self.operands[0] in ["x1", "x5"]:
            return True
        # jalr x1, rs1 where rs1 != x5 or jalr x5, rs where rs1 != x1
        if self.mnem == "jalr" and self.operands[0] in ["x1", "x5"] and (self.operands[0], self.operands[1]) not in [("x1", "x5"), ("x5", "x1")]:
            return True
        if self.mnem == "c.jalr" and "x5" not in self.operands:
            return True
        if self.mnem == "c.jal":
            return True
        
        return False


    def is_tail_call(self, syms: SymResolver):
        # jal x0
        if self.mnem == "jal" and self.operands[0] == "x0":
            return isinstance(self.operands[1], int) and self.operands[1] in syms.fn_starts

        # c.j
        if self.mnem == "c.j":
            return isinstance(self.operands[0], int) and self.operands[0] in syms.fn_starts


        # jalr x0, rs where rs != x1 and rs != x5
        if self.mnem == "jalr" and self.operands[0] == "x0" and self.operands[1] not in ["x1", "x5"]:
            return True

        # c.jr rs1 where rs1 != x1 and rs1 != x5
        if self.mnem == "c.jr" and self.operands[0] not in ["x1", "x5"]:
            return isinstance(self.operands[0], int) and self.operands[0] in syms.fn_starts
        
        # jalr t1, t3 --> jalr x6, x28
        if self.mnem == "jalr" and self.operands[0] == "x6" and self.operands[1] == "x28":
            return syms.lookup_sym(self.pc) == NOSYM_NAME

        return False

    def is_inferrable_jump(self):
        if self.mnem in ["jal", "c.jal", "c.j"] or (self.mnem == "jalr" and self.operands[1] == "x0"):
            return True
        return False

    def is_uninferrable_jump(self):
        if (self.mnem == "jalr" and self.operands[1] != "x0") or self.mnem in ["c.jalr", "c.jr"]:
            return True
        return False

    def is_return(self) -> bool:

        # ret pseudo-instruction
        if self.mnem == "ret":
            return True

        # jalr x0, 0(ra) - explicit return
        if self.mnem == "jalr" and self.operands[1] in ["x1", "x5"] and self.operands[0] not in ["x1", "x5"]:
            return True
        
        if self.mnem == "c.jr" and self.operands[0] == "x1":
            return True
        return False


def get_symbol_die(dwarfinfo, symbol_name):
    # Option A: via the pubnames lookup table
    pubnames = dwarfinfo.get_pubnames()
    if pubnames and symbol_name in pubnames:
        lut_entry = pubnames[symbol_name]
        return dwarfinfo.get_DIE_from_lut_entry(lut_entry)

    # Option B: brute‐force scan all CUs and DIEs
    for cu in dwarfinfo.iter_CUs():
        for die in cu.iter_DIEs():
            name_attr = die.attributes.get('DW_AT_name')
            if name_attr:
                name = name_attr.value.decode('utf-8', 'replace')
                if name == symbol_name:
                    return die
    return None

def find_by_mangled_name(dwarfinfo, mangled):
    """Scan every DIE’s DW_AT_linkage_name for an exact match."""
    for cu in dwarfinfo.iter_CUs():
        for die in cu.iter_DIEs():
            link = die.attributes.get('DW_AT_linkage_name') or die.attributes.get('DW_AT_MIPS_linkage_name')
            if link and link.value.decode('utf-8') == mangled:
                return die
    return None

def resolve_inlined(die, dwarfinfo):
    """If die is inlined, follow DW_AT_abstract_origin back to the real template DIE."""
    origin = die.attributes.get('DW_AT_abstract_origin')
    return dwarfinfo.get_DIE_from_refaddr(origin.value) if origin else die

def get_sym_debug_info(dwarfinfo, sym_name: str, mangled_query: str = None):

    # 1. plain-name lookup
    die = get_symbol_die(dwarfinfo, sym_name)

    # 2. mangled lookup
    if die is None and mangled_query:
        die = find_by_mangled_name(dwarfinfo, mangled_query)

    # 3. inline origin
    if die and die.tag == 'DW_TAG_inlined_subroutine':
        die = resolve_inlined(die, dwarfinfo)

    if not die:
        return None
    info = {}
    attrs = die.attributes

    # Address range for subprogram
    low_pc = attrs.get('DW_AT_low_pc')
    high_pc = attrs.get('DW_AT_high_pc')
    if low_pc and high_pc:
        info['address_range'] = (hex(low_pc.value), hex(high_pc.value))

    # Linkage name
    linkage = attrs.get('DW_AT_linkage_name') or attrs.get('DW_AT_MIPS_linkage_name')
    if linkage:
        info['linkage_name'] = linkage.value.decode('utf-8')

    # Source file + line (using CU’s line table)
    cu = die.cu
    try:
        lp = dwarfinfo.line_program_for_CU(cu)
        for entry in lp.get_entries():
            st = entry.state
            if st and not st.end_sequence and low_pc and st.address == low_pc.value:
                fe = lp['file_entry'][st.file - 1]
                info['source_file'] = fe.name.decode('utf-8')
                info['source_line'] = st.line
                break
    except Exception:
        pass

    # Type info
    if 'DW_AT_type' in attrs:
        type_offset = attrs['DW_AT_type'].value
        type_die = dwarfinfo.get_DIE_from_refaddr(type_offset)
        info['type_tag'] = type_die.tag

    # Variable location
    if die.tag == 'DW_TAG_variable':
        loc = attrs.get('DW_AT_location')
        if loc:
            info['location_expr'] = loc.value

    # Declaration site
    df = attrs.get('DW_AT_decl_file')
    dl = attrs.get('DW_AT_decl_line')
    if df and dl:
        info['decl_file'] = df.value
        info['decl_line'] = dl.value

    info['keys'] = ",".join(list(attrs.keys()))
    # Other attributes you can pull:
    #  - DW_AT_decl_file / DW_AT_decl_line
    #  - DW_AT_frame_base
    #  - DW_AT_inline (for inlining hints)
    #  - DW_AT_visibility, DW_AT_accessibility
    #  - For C++: DW_AT_virtuality, DW_AT_template_parameter
    #  - From the CU: DW_AT_producer, DW_AT_language, DW_AT_comp_dir
    return info

class TraceGen:
    def __init__(self, res_dir: Path,
                cfg: Dict,
                syms: SymResolver, 
                fn_target: Optional[str] = None, 
                skip_fns: Optional[List[str]] = None, 
                debug: bool = False,
                logger = None
                ):
        self.cfg = cfg
        self.syms = syms
        self.fn_target = fn_target
        self.retires_file = res_dir / "retires.log"
        self.trace_file = res_dir / "trace.json"
        assert self.retires_file.exists(), f"{self.retires_file} does not exist. `perf::pilos::sch::knob_enable_retire_file` must be enabled to generate a perf file."

        self.events = []
        self.call_stack = []
        self.call_starts = []
        self.prev_fn = None
        self.prev_instr = Instruction.none_instr()
        self.trace_active = False if fn_target is not None else True
        self.skip_fns = skip_fns or []
        self.debug = debug
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if len(self.logger.handlers) == 0:
                stream_handler = logging.StreamHandler(sys.stdout)
                stream_handler.setFormatter(logging.Formatter("%(message)s"))
                self.logger.addHandler(stream_handler)
                self.logger.propagate = False

    @property
    def stack_str(self):
        return "/".join([i.fn for i in self.call_stack])

    def process_instr(self, instr: Instruction, fn: str):
        if fn != self.prev_fn and self.prev_instr.cat in ["call", "tail"]:
            self.call_starts.append(instr)

        if instr.is_call():
            stack_str = self.stack_str
            self.call_stack.append(instr)
            if self.debug:
                print(f"call: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
            instr_cat = "call"
        elif instr.is_tail_call(self.syms):
            stack_str = self.stack_str
            assert self.call_starts
            # Tail calls are basically returns from the previous function
            start_instr = self.call_starts.pop()
            event = {
                "name": fn,
                "cat": "function",
                "ph": "X",  # Complete event
                "ts": start_instr.cycle,
                "dur": instr.cycle - start_instr.cycle,
                "pid": instr.pid,
                "tid": instr.tid,
                "args": {
                    "call_stack": stack_str,
                    "fn_addr": start_instr.pc,
                    "caller_addr": self.call_stack[-1].pc if self.call_stack else -1
                }
            }
            self.events.append(event)
            if self.debug:
                print(f"tail call: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
            instr_cat = "tail"
        elif instr.is_return():
            stack_str = self.stack_str
            assert self.call_starts
            start_instr = self.call_starts.pop()
            event = {
                "name": fn,
                "cat": "function",
                "ph": "X",  # Complete event
                "ts": start_instr.cycle,
                "dur": instr.cycle - start_instr.cycle,
                "pid": instr.pid,
                "tid": instr.tid,
                "args": {
                    "call_stack": stack_str,
                    "fn_addr": start_instr.pc,
                    "caller_addr": self.call_stack[-1].pc if self.call_stack else -1
                }
            }
            self.events.append(event)

            if self.call_stack:
                self.call_stack.pop()
                
            if fn == self.fn_target:
                self.trace_active = False

            if self.debug:
                print(f"return: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
            instr_cat = "return"
        else:
            stack_str = self.stack_str
            if self.debug:
                if self.prev_fn != fn and self.prev_instr.cat == "instr":
                    print(f"unidentified jump instr: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
                elif self.prev_instr.cat in ["call", "tail"] and self.prev_fn == fn and fn != NOSYM_NAME:
                    print(f"invalid function call: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
                else:
                    print(f"instr: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
            elif self.prev_fn != fn and self.prev_instr.cat == "instr":
                raise RuntimeError(f"Unidentified call instr: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")
            elif self.prev_instr.cat in ["call", "tail"] and self.prev_fn == fn and fn != NOSYM_NAME:
                raise RuntimeError(f"Incorrectly identified tail or fn call: pc={hex(instr.pc)}, fn={fn}, instr={instr}, stack={stack_str}")

            instr_cat = "instr"
        self.prev_instr = instr
        self.prev_instr.cat = instr_cat
        self.prev_fn = fn


    def run_trace(self):
        self.logger.info(f"Starting trace generation for {self.retires_file}.")
        if self.fn_target:
            self.logger.info(f"Tracing will begin when calls to {self.fn_target} are encountered.")
        with open(self.retires_file) as f:
            for line in f:
                instr = Instruction.from_str(line)
                if not instr:
                    continue
                fn = self.syms.lookup_sym(instr.pc)
                instr.fn = fn
                if not self.trace_active:
                    if fn == self.fn_target:
                        self.trace_active = True
                        self.prev_fn = None
                        # Emulate a function call to trigger a start event
                        self.prev_instr = instr
                        self.prev_instr.cat = "call"
                    else:
                        self.prev_instr = instr
                        continue
                self.process_instr(instr, fn)
        self.logger.info(f"Finished trace generation for {self.retires_file}. Number of traced events: {len(self.events)}.")


    def filter_events(self):
        filt_events = []
        for e in self.events:
            if e["name"] in self.skip_fns:
                continue
            stack = set(e["args"]["call_stack"].split("/"))
            if not stack.isdisjoint(set(self.skip_fns)):
                continue
            
            filt_events.append(e)
        return filt_events

    def write_trace_file(self):
        # wrap in Chrome trace format:
        self.logger.info(f"Filtering events from the following function calls and all children of these function calls: {self.skip_fns}")
        total_events = len(self.events)
        events = self.filter_events()
        if self.syms.debug_info:
            events = self.syms.add_debug_info(events)
        self.logger.info(f"Filtered {total_events - len(events)} events from output.")
        trace = {
            "traceEvents": events,
            "displayTimeUnit": "cycles"
        }
        with open(self.trace_file, 'w') as o:
            json.dump(trace, o, indent=2)
        self.logger.info(f"Trace file has been written to {self.trace_file}")

def get_sim_cfg(cfg_file: Path):
    with open(str(cfg_file), "r") as f:
        cfg_lines = f.readlines()
    cfg = {}
    for l in cfg_lines:
        if not l or len(l.strip()) == 0:
            continue
        if l.lstrip()[0] == "#":
            continue
        parts =  l.lstrip().split("=")
        key = parts[0].strip().replace("\n", "").replace("\r", "")
        value = "".join(parts[1:]).lstrip().replace("\n", "").replace("\r", "")
        cfg[key] = value
    # For convenience, need to also identify the binary being ran so we can create a symbol table
    assert "func::imi_spike::no_args::m_pk" in cfg, f"Expected config value `func::imi_spike::no_args::m_pk`, but none was found in {cfg_file}"
    pk_parts = cfg["func::imi_spike::no_args::m_pk"].split()

    # WARNING: this is a hacky solution to identifying the binary used in simulation. It assumes that the path to the binary will be the value directly after the path to the "pk" binary
    for i in range(len(pk_parts)):
        p = pk_parts[i]
        if "/pk" in p:
            pk_path = Path(p)
            assert len(pk_parts) > (i + 1), f"No values are defined after the pk binary path"
            bin_path = Path(pk_parts[i + 1])
            assert bin_path.exists(), f"Binary path used for execution does not exist: {bin_path}"
            cfg["bin_path"] = str(bin_path)
            break
    if "bin_path" not in cfg:
        raise ValueError(f"Unable to identify binary used for simulation with config {cfg_file}")
    return cfg

def gen_perf_file(res_dir: Path, fn_target: Optional[str] = None, debug: bool = False, logger = None):

    retires_file = res_dir / "retires.log"
    cfg_file = res_dir / "out_config.txt"
    trace_file = res_dir / "trace.json"
    assert retires_file.exists(), f"{retires_file} does not exist. `perf::pilos::sch::knob_enable_retire_file` must be enabled to generate a perf file."  
    assert cfg_file.exists(), f"{cfg_file} does not exist for simulation in {res_dir}."
    cfg = get_sim_cfg(cfg_file)
    syms = SymResolver.from_file(Path(cfg['bin_path']))
    tgen = TraceGen(res_dir, cfg, syms, fn_target=fn_target, debug=debug, skip_fns=SKIP_FUNCS, logger=logger)
    tgen.run_trace()
    tgen.write_trace_file()


def generate_profile(res_dir: Path, fn_target: Optional[str] = None, debug: bool = False, logger = None):
    res_dir = Path(res_dir) if isinstance(res_dir, str) else res_dir
    assert res_dir.exists(), f"Results directory {res_dir} does not exist!"

    sp_file = res_dir / "simpoints"
    if sp_file.exists():
        for d in res_dir.iterdir():
            if d.is_dir() and d.stem[:2] == "sp":
                gen_perf_file(d, fn_target=fn_target, debug=debug, logger=logger)
    else:
        gen_perf_file(res_dir, fn_target=fn_target, debug=debug, logger=logger)

def main():
    parser = argparse.ArgumentParser(
        description='Generate Chrome event trace from simulator instruction traces'
    )
    parser.add_argument("-r", "--result-dir", required=True, help="Specify the directory to be processed.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug info.")
    parser.add_argument("-f", "--fn-target", default=None, help="Function to profile.")
    
    args = parser.parse_args()
    generate_profile(args.result_dir, fn_target=args.fn_target, debug=args.debug)
    

if __name__ == "__main__":
    main()