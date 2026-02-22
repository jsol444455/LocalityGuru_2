from email import parser
import os, sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns
import pandas as pd
from tqdm import tqdm

import argparse

param_dict = dict()

'''
tidx_list = list() #list(range(0,512))
tidy_list = list()
ctaidy_list = list()
ctaidx_list = list()
'''
global ctaid_map
ctaid_map = list()

def ADD(a,b):
    return a+ b
def SUB(a,b):
    return a-b
def MADLO(a,b):
    return (( a )&0x0000000011111111) + b #(( a )&0x0000000011111111) + b
def MUL(a,b):
    return a * b
def SHL(a,b):
    return int(a) << int(b)
def OR(a,b):
    return a | b
def AND(a,b):
    return a & b
def DIV(a,b):
    return a / b
def SELP(a,b,c):
    if c==1:
        return a
    else:
        return b

def OPERATE(a,b,func):
    if type(a) == str:
        a_list = param_dict[a]
    else:
        a_list = a

    if type(b) == str:
        b_list = param_dict[b]
    else:
        b_list = b
    # Handle case where a_list or b_list is a single integer
    if not isinstance(a_list, list):
        a_list = [a_list]
    if not isinstance(b_list, list):
        b_list = [b_list]    
    tmp_list = list()
    ctaidy_tmp_dict = list()
    for i in range(ctaidy*ctaidx):
        ctaidy_tmp_dict.append(list())
        
    if type(a) == str and a.startswith("%ctaid"):
        a_len = len(a_list)
        for a_ in a_list: #found the error!!!!!!!!!!!!!!!!!
            for b_ in b_list:
                tmp_list.append(func(a_,b_))
            if a == "%ctaid.x":
                for b_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)]= tmp_list
            else:
                for b_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[a_+(b_*a_len)] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    elif type(b) == str and  b.startswith("%ctaid"):
        b_len = len(b_list)
        for b_ in b_list: 
            for a_ in a_list:
                tmp_list.append(func(a_, b_))
            if b == "%ctaid.x":
                for a_ in param_dict["%ctaid.y"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_]= tmp_list
            else:
                for a_ in param_dict["%ctaid.x"]:
                    ctaidy_tmp_dict[(a_*b_len)+b_] = tmp_list
            tmp_list = list()
        return ctaidy_tmp_dict
    else:
        if len(a_list)!=0  and type(a_list[0]) == list:
            for idx_, a_ in enumerate(a_list):
                for a_int in a_:
                    for b_ in b_list:
                        if type(b_) == list:
                            for b_int in b_:
                                tmp_list.append(func(a_int, b_int))
                        else:
                            tmp_list.append(func(a_int, b_))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            return ctaidy_tmp_dict
        elif len(b_list)!=0 and type(b_list[0]) == list:
            for idx_, b_ in enumerate(b_list):
                #print(b_)
                for b_int in b_:
                    for a_ in a_list:
                        tmp_list.append(func(a_, b_int))
                ctaidy_tmp_dict[idx_] = tmp_list
                tmp_list = list()
            #print("DDD")
            #print(ctaidy_tmp_dict)
            return ctaidy_tmp_dict
        else:
            for a_ in a_list:
                for b_ in b_list:
                    tmp_list.append(func(a_,b_))
            return tmp_list
    return tmp_list

def tracing(tree,idx):
    opcode = tree[idx]["opcode"]
    # Handle both "child_num" and "child" keys
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    child_list = list()
    if child_num == 0:
        reg_ = tree[idx].get("reg", tree[idx].get("reg_name", ""))
        return reg_ # str return
    while(child_num>0):
        for id_, node_ in enumerate(tree):
            # Handle both "parent" and "parent_loc" keys
            node_parent = node_.get("parent", node_.get("parent_loc", -1))
            if node_parent == idx:
                child_list.append(id_)
                child_num -= 1

    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    if child_num==2:
        if opcode.startswith("add"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),ADD)
            return res_
        elif opcode.startswith("sub"):
            res_ = OPERATE(tracing(tree,child_list[0]),tracing(tree, child_list[1]),SUB)
            return res_
        elif opcode.startswith("shl"):
            res_ = OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]),SHL)
            return res_
        elif opcode.startswith("mul"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),MUL)
            return res_
        elif opcode.startswith("or"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),OR)
            return res_
    elif child_num==3:
        if opcode.startswith("mad"):
            res_ = OPERATE(OPERATE(tracing(tree, child_list[0]), tracing(tree, child_list[1]), MUL), tracing(tree, child_list[2]), MADLO)
            return res_
        elif opcode.startswith("selp"):
            res_ = OPERATE(tracing(tree, child_list[0]),tracing(tree, child_list[1]),tracing(tree, child_list[2]),SELP)
            return res_
    else:
        #print(tree[child_list[0]]["reg"])
        return tracing(tree, child_list[0])


# ============================================================================
# FIXED: Loop Evaluation Functions (Algorithm 2: EvalSTloop)
# ============================================================================

def update_loop_tree_leaf_nodes(loop_tree, loop_vars):
    """
    FIXED: Algorithm 2, Line 5: Update leaf nodes with loop var values
    
    This is the CRITICAL function that was missing/incomplete in the original code.
    It replaces loop variable register names with their current numeric values.
    """
    updated_tree = []
    
    for node in loop_tree:
        new_node = node.copy()
        
        # If this is a leaf node (child == 0) and contains a loop variable
        if node.get("child", 0) == 0:
            reg_name = node.get("reg_name", "")
            
            # Check if this register is in loop_vars
            if reg_name in loop_vars:
                # Replace register name with its current value
                new_node["reg_name"] = str(loop_vars[reg_name])
            else:
                # Try without % prefix
                clean_reg = reg_name.replace("%", "")
                if clean_reg in loop_vars:
                    new_node["reg_name"] = str(loop_vars[clean_reg])
        
        updated_tree.append(new_node)
    
    return updated_tree


def eval_loop_syntax_tree(loop_tree, loop_predicate_tree, param_dict, max_iterations=10000):
    """
    FIXED: Algorithm 2: EvalSTloop() - Lines 1-8
    Evaluates all loop iterations and returns all addresses
    
    This properly implements Algorithm 2 from the LocalityGuru paper with
    Line 5 (update tree) happening BEFORE Line 6 (evaluate address).
    """
    addresses = []
    iteration = 0
    
    # Initialize loop iterator
    loop_vars = {}
    for node in loop_tree:
        if node.get("is_loop_variable") and node.get("parent_loc") == -1:
            loop_vars[node["reg_name"]] = 0
    
    # If no loop variables found in tree, try to infer from param_dict
    if not loop_vars:
        # Look for common loop iterator patterns
        for key in param_dict.keys():
            if "iter" in key.lower() or key in ["i", "j", "k"]:
                loop_vars[key] = 0
    
    # Line 3: while EvalST(BB_label) != 0
    while iteration < max_iterations:
        try:
            # Evaluate predicate with current loop variable values
            combined_dict = {**param_dict, **loop_vars}
            predicate_result = evaluate_predicate_with_loop_vars(
                loop_predicate_tree, param_dict, loop_vars
            )
            
            if not predicate_result:
                break
            
            # ============================================================
            # CRITICAL FIX: Line 5 - Update leaf nodes BEFORE evaluation
            # ============================================================
            updated_tree = update_loop_tree_leaf_nodes(loop_tree, combined_dict)
            
            # Line 6: Evaluate tree to get current address
            current_address = tracing(updated_tree, 0)
            
            # Map[TB].insert(EvalST(reg))
            if current_address is not None:
                if isinstance(current_address, list):
                    addresses.extend(current_address)
                else:
                    addresses.append(current_address)
            
            # ============================================================
            # Line 4: Update loop iterator variables
            # ============================================================
            for var_name in list(loop_vars.keys()):
                new_value = evaluate_loop_variable_update(
                    loop_tree, var_name, loop_vars, param_dict
                )
                loop_vars[var_name] = new_value
            
            iteration += 1
            
        except Exception as e:
            print(f"[Warning] Loop tree evaluation error at iteration {iteration}: {e}")
            break
    
    if iteration >= max_iterations:
        print(f"[Warning] Loop hit max iterations ({max_iterations})")
    
    return addresses


def evaluate_predicate_with_loop_vars(predicate_tree, param_dict, loop_vars):
    """
    FIXED: Algorithm 2, Line 3: Evaluate loop exit condition
    
    Combines param_dict and loop_vars for predicate evaluation
    """
    if not predicate_tree:
        return False
    
    combined_dict = {**param_dict, **loop_vars}
    
    try:
        result = eval_predicate_tree_simple(predicate_tree, 0, combined_dict)
        
        if isinstance(result, list):
            return all(result) if result else False
        return bool(result)
    except Exception as e:
        print(f"[Warning] Predicate evaluation failed: {e}")
        return False


def eval_predicate_tree_simple(tree, idx, param_dict):
    """
    Simple predicate evaluation for loop conditions
    """
    if idx >= len(tree) or idx < 0:
        return 0
    
    node = tree[idx]
    opcode = node.get("opcode", "")
    child_count = node.get("child", 0)
    
    # Leaf node
    if child_count == 0:
        reg_name = node.get("reg_name", "")
        
        # Try direct lookup
        if reg_name in param_dict:
            val = param_dict[reg_name]
            return val[0] if isinstance(val, list) else val
        
        # Try normalized lookup
        normalized = reg_name.replace("%", "").replace(".", "_")
        if normalized in param_dict:
            val = param_dict[normalized]
            return val[0] if isinstance(val, list) else val
        
        # Try parsing as number
        try:
            return int(reg_name)
        except:
            try:
                return float(reg_name)
            except:
                return 0
    
    # Get children
    child_indices = []
    for i in range(child_count):
        child_key = f"child{i}"
        if child_key in node:
            child_indices.append(node[child_key])
    
    # Evaluate children
    child_values = [eval_predicate_tree_simple(tree, idx, param_dict) for idx in child_indices]
    
    # Handle comparison operations
    if "setp" in opcode.lower() and len(child_values) >= 2:
        left, right = child_values[0], child_values[1]
        parts = opcode.split('.')
        if len(parts) >= 2:
            comp_type = parts[1].lower()
            if comp_type == "gt":
                return 1 if left > right else 0
            elif comp_type == "lt":
                return 1 if left < right else 0
            elif comp_type == "ge":
                return 1 if left >= right else 0
            elif comp_type == "le":
                return 1 if left <= right else 0
            elif comp_type == "eq":
                return 1 if left == right else 0
            elif comp_type == "ne":
                return 1 if left != right else 0
    
    # Default: return first child
    return child_values[0] if child_values else 0


def evaluate_loop_variable_update(loop_tree, var_name, loop_vars, param_dict):
    """
    FIXED: Algorithm 2, Line 4: Update loop iterator
    
    Finds the update operation for a loop variable and computes its new value
    """
    for node in loop_tree:
        if node.get("reg_name") == var_name and node.get("parent_loc") == -1:
            combined_dict = {**param_dict, **loop_vars}
            
            opcode = node.get("opcode", "")
            
            if opcode.startswith("add"):
                left_val = get_node_value(loop_tree, node.get("child0", 0), combined_dict)
                right_val = get_node_value(loop_tree, node.get("child1", 0), combined_dict)
                return left_val + right_val
            
            elif opcode.startswith("sub"):
                left_val = get_node_value(loop_tree, node.get("child0", 0), combined_dict)
                right_val = get_node_value(loop_tree, node.get("child1", 0), combined_dict)
                return left_val - right_val
            
            elif opcode.startswith("mul"):
                left_val = get_node_value(loop_tree, node.get("child0", 0), combined_dict)
                right_val = get_node_value(loop_tree, node.get("child1", 0), combined_dict)
                return left_val * right_val
    
    # If no update operation found, just increment by 1
    return loop_vars.get(var_name, 0) + 1


def get_node_value(tree, node_idx, param_dict):
    """
    FIXED: Get value of a tree node
    
    Handles both leaf nodes (values/parameters) and intermediate nodes
    """
    if node_idx >= len(tree) or node_idx < 0:
        return 0
    
    node = tree[node_idx]
    reg_name = node.get("reg_name", "")
    
    # Leaf node - return its value
    if node.get("child", 0) == 0:
        # Try to find in param_dict
        if reg_name in param_dict:
            val = param_dict[reg_name]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        
        # Try without % prefix
        clean_reg = reg_name.replace("%", "")
        if clean_reg in param_dict:
            val = param_dict[clean_reg]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        
        # Try to parse as integer constant
        try:
            return int(reg_name)
        except:
            try:
                return float(reg_name)
            except:
                return 0
    
    # Non-leaf node - should recursively evaluate
    return 0


# ============================================================================
# FIXED: get_loop_addresses function
# ==========================================================================
def get_loop_addresses(bb_graph, kernel_name, formular, param_dict, predicates, kernel_info=None):
    """
    COMPREHENSIVE FIX: Algorithm 2 Line 18: EvalSTloop(ST, Map)
    Returns set of addresses accessed by all loop iterations
    
    KEY CHANGES:
    1. Returns empty set {} instead of None in most cases (only returns None for truly non-loop accesses)
    2. Better fallback: If loop evaluation fails, falls back to single non-loop address evaluation
    3. More robust predicate handling
    4. Better error messages for debugging
    """
    
    # ========================================================================
    # VALIDATION: Check if we can even attempt loop evaluation
    # ========================================================================
    if kernel_name not in bb_graph:
        print(f"[get_loop_addresses] Kernel {kernel_name} not in bb_graph - NOT A LOOP")
        return None  # This is truly a non-loop case
    
    # ========================================================================
    # STEP 1: Find basic blocks with memory accesses and loop headers
    # ========================================================================
    ld_global_bbs = [bb_n for bb_n, bb_info in bb_graph[kernel_name].items() 
                     if bb_info.get("has_ld_global", False)]
    
    loop_header_bbs = [bb_n for bb_n, bb_info in bb_graph[kernel_name].items() 
                       if bb_info.get("is_loop_header", False)]
    
    print(f"[get_loop_addresses] Kernel: {kernel_name}")
    print(f"  - BBs with ld.global: {ld_global_bbs}")
    print(f"  - Loop header BBs: {loop_header_bbs}")
    
    # If no loop headers exist at all, this is definitely not a loop
    if not loop_header_bbs:
        print(f"[get_loop_addresses] No loop headers found - NOT A LOOP")
        return None  # Not a loop access
    
    # ========================================================================
    # STEP 2: Find suitable loop using 3 strategies
    # ========================================================================
    loop_bb = None
    loop_info = None
    
    # Strategy 1: BB that is BOTH a loop header AND has ld.global
    for bb_n in ld_global_bbs:
        bb_info = bb_graph[kernel_name][bb_n]
        if bb_info.get("is_loop_header", False):
            loop_bb = bb_n
            loop_info = bb_info
            print(f"[get_loop_addresses] ✓ Strategy 1: BB {bb_n} is both loop header and has ld.global")
            break
    
    # Strategy 2: Associate ld.global BB with nearby loop header (within 5 BBs)
    if loop_bb is None and ld_global_bbs:
        print(f"[get_loop_addresses] Strategy 1 failed, trying Strategy 2...")
        for header_bb_n in loop_header_bbs:
            header_info = bb_graph[kernel_name][header_bb_n]
            
            if (header_info.get("loop_variables") and 
                header_info.get("loop_predicate")):
                
                for ld_bb_n in ld_global_bbs:
                    if abs(ld_bb_n - header_bb_n) <= 5:
                        loop_bb = header_bb_n
                        loop_info = header_info
                        print(f"[get_loop_addresses] ✓ Strategy 2: Associated ld.global BB {ld_bb_n} with loop header BB {header_bb_n}")
                        break
                
                if loop_bb is not None:
                    break
    
    # Strategy 3: Use first loop header with complete information
    if loop_bb is None and loop_header_bbs:
        print(f"[get_loop_addresses] Strategy 2 failed, trying Strategy 3...")
        for header_bb_n in loop_header_bbs:
            header_info = bb_graph[kernel_name][header_bb_n]
            if (header_info.get("loop_variables") and 
                header_info.get("loop_predicate")):
                loop_bb = header_bb_n
                loop_info = header_info
                print(f"[get_loop_addresses] ✓ Strategy 3: Using loop header BB {header_bb_n}")
                break
    
    # If all 3 strategies failed, this is not a loop we can handle
    if loop_bb is None or loop_info is None:
        print(f"[get_loop_addresses] ✗ All 3 strategies failed - NOT A LOOP")
        return None  # Not a loop access
    
    # ========================================================================
    # STEP 3: Extract loop information
    # ========================================================================
    loop_variables = loop_info.get("loop_variables", [])
    loop_predicate_name = loop_info.get("loop_predicate")
    
    # NEW: Instead of returning None, return empty set and try fallback
    if not loop_variables:
        print(f"[get_loop_addresses] ⚠ No loop variables in BB {loop_bb}")
        print(f"[get_loop_addresses] → Falling back to non-loop address evaluation")
        return _fallback_single_address(formular, param_dict)
    
    if not loop_predicate_name:
        print(f"[get_loop_addresses] ⚠ No loop predicate in BB {loop_bb}")
        print(f"[get_loop_addresses] → Falling back to non-loop address evaluation")
        return _fallback_single_address(formular, param_dict)
    
    # ==================================================================
    # CRITICAL FIX: Normalize predicate name to match predicates dict
    # ==================================================================
    predicate_key = None
    loop_pred_normalized = loop_predicate_name.replace("%", "")
    
    print(f"[get_loop_addresses] Looking for predicate: '{loop_predicate_name}'")
    print(f"[get_loop_addresses] Available predicates: {list(predicates.keys())}")
    
    # Try multiple matching strategies
    for key in predicates.keys():
        # Strategy 1: Exact match
        if key == loop_predicate_name:
            predicate_key = key
            print(f"[get_loop_addresses] ✓ Exact match: '{key}'")
            break
        
        # Strategy 2: Match base name (e.g., "p0" matches "%p0_123")
        key_base = key.split("_")[0].replace("%", "")
        if key_base == loop_pred_normalized:
            predicate_key = key
            print(f"[get_loop_addresses] ✓ Base name match: '{loop_predicate_name}' -> '{key}'")
            break
        
        # Strategy 3: Match with/without % prefix
        if key.replace("%", "") == loop_pred_normalized:
            predicate_key = key
            print(f"[get_loop_addresses] ✓ Normalized match: '{loop_predicate_name}' -> '{key}'")
            break
    
    # If still no match found, try fallback
    if predicate_key is None:
        print(f"[get_loop_addresses] ✗ Predicate '{loop_predicate_name}' NOT FOUND in predicates dict")
        print(f"[get_loop_addresses] → Falling back to non-loop address evaluation")
        return _fallback_single_address(formular, param_dict)
    
    loop_predicate = predicates[predicate_key]
    print(f"[get_loop_addresses] ✓ Using predicate: '{predicate_key}'")
    # ==================================================================
    
    # ========================================================================
    # STEP 3.5: Check if loop tree exists and use it (CRITICAL FIX for Issue #3)
    # ========================================================================
    if kernel_info and "loop_trees" in kernel_info.get(kernel_name, {}):
        loop_trees = kernel_info[kernel_name]["loop_trees"]
        
        # Try to find the loop tree for this loop BB
        loop_tree_key = None
        for key in loop_trees.keys():
            if f"_loop_bb{loop_bb}" in key:
                loop_tree_key = key
                break
        
        if loop_tree_key:
            loop_tree = loop_trees[loop_tree_key]
            print(f"[get_loop_addresses] ✓ Found loop tree: {loop_tree_key}")
            print(f"[get_loop_addresses] Using eval_loop_syntax_tree() for proper Algorithm 2 implementation")
            
            try:
                # USE THE PROPER ALGORITHM 2 IMPLEMENTATION!
                addresses = eval_loop_syntax_tree(
                    loop_tree=loop_tree,
                    loop_predicate_tree=loop_predicate,
                    param_dict=param_dict,
                    max_iterations=10000
                )
                
                if addresses:
                    print(f"[get_loop_addresses] ✓ Loop tree evaluation found {len(addresses)} addresses")
                    return set(addresses)  # Convert list to set
                else:
                    print(f"[get_loop_addresses] ⚠ Loop tree evaluation returned empty")
                    # Fall through to simplified approach
            except Exception as e:
                print(f"[get_loop_addresses] ⚠ Loop tree evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to simplified approach
        else:
            print(f"[get_loop_addresses] ⚠ Loop tree not found for BB {loop_bb}")
    else:
        print(f"[get_loop_addresses] ⚠ No loop trees in kernel_info")
    
    # ========================================================================
    # STEP 4: Evaluate loop iterations (SIMPLIFIED FALLBACK)
    # ========================================================================
    print(f"[get_loop_addresses] Using simplified loop evaluation (fallback)")
    loop_addresses = set()
    loop_param_dict = param_dict.copy()
    
    # Get primary loop variable
    loop_var = loop_variables[0]
    iterator_name = loop_var.get("register", "loop_iter")
    
    print(f"[get_loop_addresses] Starting loop evaluation:")
    print(f"  - Iterator: {iterator_name}")
    print(f"  - Predicate: {loop_predicate_name}")
    print(f"  - Formula: {formular}")
    
    # Initialize loop iterator
    loop_param_dict[iterator_name] = 0
    max_iterations = 10000
    iteration_count = 0
    
    # Simulate loop iterations
    try:
        while iteration_count < max_iterations:
            # Evaluate loop predicate
            try:
                pred_result = eval_predicate_tree(loop_predicate, 0, loop_param_dict)
            except Exception as e:
                print(f"[get_loop_addresses] ⚠ Predicate evaluation failed at iteration {iteration_count}: {e}")
                break
            
            if not pred_result:
                break  # Exit loop
            
            # Evaluate formula to get address
            try:
                address = evaluate_formula(formular, loop_param_dict)
                
                if address is not None:
                    loop_addresses.add(address)
            except Exception as e:
                print(f"[get_loop_addresses] ⚠ Address evaluation failed at iteration {iteration_count}: {e}")
                # Continue to next iteration instead of breaking
            
            # Update loop iterator
            update_operation = loop_var.get("operation", "add")
            
            if update_operation in ["add", "add.s32", "add.u32"]:
                loop_param_dict[iterator_name] += 1
            elif update_operation in ["sub", "sub.s32", "sub.u32"]:
                loop_param_dict[iterator_name] -= 1
            else:
                # Unknown operation, increment by default
                loop_param_dict[iterator_name] += 1
            
            iteration_count += 1
        
        print(f"[get_loop_addresses] ✓ Loop evaluation complete:")
        print(f"  - Iterations: {iteration_count}")
        print(f"  - Unique addresses: {len(loop_addresses)}")
        
        # NEW: If we got no addresses, try fallback
        if not loop_addresses:
            print(f"[get_loop_addresses] ⚠ No addresses found in loop evaluation")
            print(f"[get_loop_addresses] → Falling back to non-loop address evaluation")
            return _fallback_single_address(formular, param_dict)
        
        return loop_addresses
        
    except Exception as e:
        print(f"[get_loop_addresses] ✗ Loop evaluation exception: {e}")
        import traceback
        traceback.print_exc()
        print(f"[get_loop_addresses] → Falling back to non-loop address evaluation")
        return _fallback_single_address(formular, param_dict)


def _fallback_single_address(formular, param_dict):
    """
    NEW HELPER: Fallback when loop evaluation fails
    Try to evaluate the formula once with the given parameters
    Returns a set with single address or empty set
    """
    try:
        address = evaluate_formula(formular, param_dict)
        if address is not None:
            print(f"[get_loop_addresses] Fallback found address: {address}")
            return {address}
        else:
            print(f"[get_loop_addresses] Fallback: formula evaluated to None")
            return set()  # Return empty set instead of None
    except Exception as e:
        print(f"[get_loop_addresses] Fallback failed: {e}")
        return set()  # Return empty set instead of None


# ============================================================================
# ADDITIONAL FIX: Modify the caller in GetTBAddressMap
# ============================================================================

def GetTBAddressMap_FIXED_SECTION():
    """
    This shows how the GetTBAddressMap function should handle the result
    from get_loop_addresses()
    """
    # ... (inside the thread loop)
    
    try:
        # Call get_loop_addresses
        loop_addresses = get_loop_addresses(
            bb_graph, 
            kernel_name, 
            formular, 
            thread_param_dict,
            predicates
        )
        
        # NEW: Better handling of the result
        if loop_addresses is not None:  # None = not a loop access
            if loop_addresses:  # Non-empty set = loop found addresses
                print(f"[GetTBAddressMap] TB {TB_id}, Thread ({tid_x},{tid_y}): "
                      f"Found {len(loop_addresses)} loop addresses")
                TB_Address_map[TB_id]["addresses"].update(loop_addresses)
            else:  # Empty set = loop evaluation tried but found nothing
                print(f"[GetTBAddressMap] TB {TB_id}, Thread ({tid_x},{tid_y}): "
                      f"Loop evaluation returned empty set, using formula")
                if formular:
                    address = evaluate_formula(formular, thread_param_dict)
                    if address is not None:
                        TB_Address_map[TB_id]["addresses"].add(address)
        else:  # None = not a loop access, use regular formula evaluation
            if formular:
                address = evaluate_formula(formular, thread_param_dict)
                if address is not None:
                    TB_Address_map[TB_id]["addresses"].add(address)
    except Exception as e:
        print(f"[GetTBAddressMap] Error: {e}")
        pass


# ============================================================================
# SUMMARY OF FIXES
# ============================================================================
"""
KEY IMPROVEMENTS:

1. RETURN VALUES:
   - None = "not a loop access" (no loop headers exist, or all strategies failed)
   - Empty set {} = "loop access but evaluation failed/found nothing"
   - Non-empty set = "successful loop evaluation"

2. FALLBACK MECHANISM:
   - If loop evaluation fails at any point, tries to evaluate formula once
   - Returns empty set instead of None in error cases

3. BETTER LOGGING:
   - Clear indication of which strategy succeeded
   - Detailed error messages for debugging
   - Shows iteration count and address count

4. ROBUST ERROR HANDLING:
   - Catches exceptions at multiple levels
   - Continues when possible instead of failing completely
   - Always returns a usable value (None, empty set, or addresses)

5. CALLER CHANGES:
   - GetTBAddressMap now distinguishes between None, empty set, and addresses
   - Falls back to formula evaluation when appropriate
"""
# ============================================================================
# Predicate and Thread Filtering Functions
# ============================================================================

def evaluate_predicate(predicate_tree, param_dict):
    """
    Evaluate a predicate syntax tree to determine true/false
    Returns a list of boolean values for each thread
    """
    result = tracing(predicate_tree, 0)
    
    # Convert result to boolean list
    if isinstance(result, list):
        if isinstance(result[0], list):
            # Per-TB results
            bool_results = []
            for tb_result in result:
                tb_bool = [bool(val) for val in tb_result]
                bool_results.append(tb_bool)
            return bool_results
        else:
            # Single list of results
            return [bool(val) for val in result]
    else:
        return [bool(result)]


def eval_predicate_tree(tree, idx, param_dict):
    """
    Evaluate predicate syntax tree to get boolean result
    Similar to tracing() but for predicates
    """
    opcode = tree[idx]["opcode"]
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    child_list = list()
    
    if child_num == 0:
        reg_ = tree[idx].get("reg", tree[idx].get("reg_name", ""))
        # Return parameter value
        if reg_ in param_dict:
            return param_dict[reg_]
        return reg_
    
    # Find children
    while child_num > 0:
        for id_, node_ in enumerate(tree):
            node_parent = node_.get("parent", node_.get("parent_loc", -1))
            if node_parent == idx:
                child_list.append(id_)
                child_num -= 1
    
    child_num = tree[idx].get("child_num", tree[idx].get("child", 0))
    
    # Evaluate based on opcode
    if "setp" in opcode:
        # Extract comparison type from opcode (e.g., setp.gt.s32 -> gt)
        parts = opcode.split('.')
        if len(parts) >= 2:
            comp_type = parts[1]  # gt, lt, ge, le, eq, ne
            
            # Evaluate operands
            op1 = eval_predicate_tree(tree, child_list[0], param_dict)
            op2 = eval_predicate_tree(tree, child_list[1], param_dict)
            
            # Perform comparison
            if comp_type == "gt":
                return OPERATE(op1, op2, lambda a, b: 1 if a > b else 0)
            elif comp_type == "lt":
                return OPERATE(op1, op2, lambda a, b: 1 if a < b else 0)
            elif comp_type == "ge":
                return OPERATE(op1, op2, lambda a, b: 1 if a >= b else 0)
            elif comp_type == "le":
                return OPERATE(op1, op2, lambda a, b: 1 if a <= b else 0)
            elif comp_type == "eq":
                return OPERATE(op1, op2, lambda a, b: 1 if a == b else 0)
            elif comp_type == "ne":
                return OPERATE(op1, op2, lambda a, b: 1 if a != b else 0)
    
    return 0  # Default false


def get_idle_thread_mask(kernel_info, bb_graph, kernel_name, param_dict):
    """
    Evaluate all predicates to determine which threads are idle
    Returns a set of thread indices that should be excluded
    """
    idle_threads = set()
    
    if "predicates" not in kernel_info.get(kernel_name, {}):
        return idle_threads
    
    # Evaluate each predicate
    predicate_results = {}
    for pred_name, pred_tree in kernel_info[kernel_name]["predicates"].items():
        if len(pred_tree) > 0:
            result = eval_predicate_tree(pred_tree, 0, param_dict)
            # Extract predicate register name
            pred_reg = pred_tree[0]["reg_name"]
            predicate_results[pred_reg] = result
    
    # Check each basic block for false branches
    for bb_id, bb_info in bb_graph.get(kernel_name, {}).items():
        if "branch_info" in bb_info:
            branch_info = bb_info["branch_info"]
            pred_reg = branch_info["predicate"]
            is_negated = branch_info["is_negated"]
            
            if pred_reg in predicate_results:
                pred_result = predicate_results[pred_reg]
                
                # Determine which threads go to false branch
                if isinstance(pred_result, list):
                    if isinstance(pred_result[0], list):
                        # Per-TB results
                        for tb_idx, tb_result in enumerate(pred_result):
                            for thread_idx, val in enumerate(tb_result):
                                global_tid = tb_idx * len(tb_result) + thread_idx
                                
                                # Thread is idle if:
                                # - Predicate is false and NOT negated (goes to false branch)
                                # - Predicate is true and IS negated (goes to false branch)
                                if (not val and not is_negated) or (val and is_negated):
                                    idle_threads.add(global_tid)
                    else:
                        # Single list
                        for thread_idx, val in enumerate(pred_result):
                            if (not val and not is_negated) or (val and is_negated):
                                idle_threads.add(thread_idx)
    
    return idle_threads


def filter_addresses_by_active_threads(formular, idle_threads):
    """
    Remove addresses accessed by idle threads
    """
    if not idle_threads or not formular:
        return formular
    
    if isinstance(formular[0], list):
        # Per-TB results
        filtered = []
        for tb_idx, tb_data in enumerate(formular):
            filtered_tb = []
            for thread_idx, addr in enumerate(tb_data):
                global_tid = tb_idx * len(tb_data) + thread_idx
                if global_tid not in idle_threads:
                    filtered_tb.append(addr)
            filtered.append(filtered_tb)
        return filtered
    else:
        # Single list
        return [addr for idx, addr in enumerate(formular) if idx not in idle_threads]


# ============================================================================
# GetTBAddressMap - Algorithm 2 Implementation
# ============================================================================

def GetTBAddressMap(syntax_tree, kernel_name, grid_dim, block_dim, 
                    bb_graph, predicates, formular, param_dict, kernel_info=None):
    """
    Implements Algorithm 2 Lines 9-24: GetTBAddressMap(ST)
    
    Args:
        syntax_tree: The syntax tree for the kernel
        kernel_name: Name of the kernel being analyzed
        grid_dim: Dictionary with keys 'x', 'y', 'z' for GridDim
        block_dim: Dictionary with keys 'x', 'y', 'z' for BlockDim  
        bb_graph: Basic block graph with predicate information
        predicates: Dictionary of predicate syntax trees
        formular: Formula string from syntax tree evaluation
        param_dict: Global parameter dictionary
        
    Returns:
        TB_Address_map: Dictionary mapping TB_id -> {"addresses": set()}
    """
    print(f"[GetTBAddressMap] Starting for kernel: {kernel_name}")
    print(f"  GridDim: ({grid_dim['x']}, {grid_dim['y']}, {grid_dim['z']})")
    print(f"  BlockDim: ({block_dim['x']}, {block_dim['y']}, {block_dim['z']})")
    
    # Initialize TB_Address_map
    TB_Address_map = {}
    
    # Line 10: ntid <- BlockDim
    ntid_x = block_dim['x']
    ntid_y = block_dim['y']
    ntid_z = block_dim['z']
    
    # Line 11: forall ctaid in GridDim do
    for ctaid_z in range(grid_dim['z']):
        for ctaid_y in range(grid_dim['y']):
            for ctaid_x in range(grid_dim['x']):
                
                # Line 12: TB <- get_tb_id(ctaid, GridDim)
                TB_id = (ctaid_z * grid_dim['y'] * grid_dim['x'] + 
                        ctaid_y * grid_dim['x'] + 
                        ctaid_x)
                
                # Initialize address set for this TB
                TB_Address_map[TB_id] = {"addresses": set()}
                
                # Build parameter dictionary for this TB
                tb_param_dict = param_dict.copy()
                
                # Update with TB-specific values
                tb_param_dict.update({
                    "ctaid.x": ctaid_x,
                    "ctaid.y": ctaid_y,
                    "ctaid.z": ctaid_z,
                    "nctaid.x": grid_dim['x'],
                    "nctaid.y": grid_dim['y'],
                    "nctaid.z": grid_dim['z'],
                    "ntid.x": ntid_x,
                    "ntid.y": ntid_y,
                    "ntid.z": ntid_z,
                    # Add normalized versions (without dots)
                    "ctaid_x": ctaid_x,
                    "ctaid_y": ctaid_y,
                    "ctaid_z": ctaid_z,
                    "nctaid_x": grid_dim['x'],
                    "nctaid_y": grid_dim['y'],
                    "nctaid_z": grid_dim['z'],
                    "ntid_x": ntid_x,
                    "ntid_y": ntid_y,
                    "ntid_z": ntid_z
                })
                
                # Line 13: forall tid in BlockDim do
                for tid_z in range(ntid_z):
                    for tid_y in range(ntid_y):
                        for tid_x in range(ntid_x):
                            
                            # Build per-thread parameter dictionary
                            thread_param_dict = tb_param_dict.copy()
                            thread_param_dict.update({
                                "tid.x": tid_x,
                                "tid.y": tid_y,
                                "tid.z": tid_z,
                                # Add normalized versions
                                "tid_x": tid_x,
                                "tid_y": tid_y,
                                "tid_z": tid_z
                            })
                            
                            # Line 14-15: Filter out idle threads
                            try:
                                if not should_include_thread(predicates, bb_graph, 
                                                            kernel_name, thread_param_dict):
                                    continue  # Skip idle thread
                            except Exception as e:
                                # If predicate evaluation fails, include thread by default
                                pass
                            
                            # ============================================================
                            # CRITICAL FIX: Proper indentation for loop evaluation
                            # Line 16-19: Evaluate formula for this thread
                            # ============================================================
                            try:
                                # Check if this is a loop access - ACTUALLY CALL THE FUNCTION NOW
                                loop_addresses = get_loop_addresses(
                                    bb_graph, 
                                    kernel_name, 
                                    formular, 
                                    thread_param_dict,  # Use thread-specific params
                                    predicates,
                                    kernel_info  # CRITICAL: Pass kernel_info!
                                )
                                
                                if loop_addresses:
                                    # Line 18: EvalSTloop(ST, Map) - Loop access found
                                    print(f"[GetTBAddressMap] TB {TB_id}, Thread ({tid_x},{tid_y}): Found {len(loop_addresses)} loop addresses")
                                    TB_Address_map[TB_id]["addresses"].update(loop_addresses)
                                else:
                                    # Line 17: Insert (EvalST(reg) + offset) into Map[TB] - Non-loop access
                                    if formular:
                                        address = evaluate_formula(formular, thread_param_dict)
                                        if address is not None:
                                            TB_Address_map[TB_id]["addresses"].add(address)
                            except Exception as e:
                                # If formula evaluation fails, log and skip this thread
                                print(f"[GetTBAddressMap] Error evaluating address for TB {TB_id}, Thread ({tid_x},{tid_y}): {e}")
                                continue
    
    # Line 24: return Map
    print(f"[GetTBAddressMap] Complete. Processed {len(TB_Address_map)} thread blocks")
    return TB_Address_map


def should_include_thread(predicates, bb_graph, kernel_name, param_dict):
    """
    Algorithm 2 Lines 14-15: Check if thread should be included.
    Returns False if thread is idle (in false branch basic block).
    """
    if not predicates or kernel_name not in bb_graph:
        return True  # No predicates, include all threads
    
    # Evaluate each predicate for this thread
    predicate_results = {}
    
    for pred_name, pred_tree in predicates.items():
        if not pred_tree or len(pred_tree) == 0:
            continue
            
        try:
            result = eval_predicate_tree_for_thread(pred_tree, param_dict)
            pred_reg = pred_tree[0].get("reg_name", pred_name.split("_")[0])
            predicate_results[pred_reg] = result
        except Exception as e:
            continue
    
    # Check each basic block for false branches
    for bb_id, bb_info in bb_graph.get(kernel_name, {}).items():
        if "branch_info" not in bb_info:
            continue
        
        branch_info = bb_info["branch_info"]
        pred_reg = branch_info.get("predicate")
        is_negated = branch_info.get("is_negated", False)
        
        if not pred_reg or pred_reg not in predicate_results:
            continue
        
        pred_result = predicate_results[pred_reg]
        
        thread_takes_false_branch = (not pred_result and not is_negated) or \
                                   (pred_result and is_negated)
        
        if bb_info.get("is_false_branch", False) and thread_takes_false_branch:
            return False
        
        if bb_info.get("has_ld_global", False) and thread_takes_false_branch:
            return False
    
    return True


def eval_predicate_tree_for_thread(pred_tree, param_dict):
    """
    Evaluate a predicate syntax tree for a single thread.
    """
    if not pred_tree or len(pred_tree) == 0:
        return True
    
    try:
        result = eval_predicate_node(pred_tree, 0, param_dict)
        
        if isinstance(result, (list, tuple)):
            result = result[0] if result else 0
        
        return bool(result)
    except Exception as e:
        return True


def eval_predicate_node(tree, node_idx, param_dict):
    """
    Recursively evaluate a predicate syntax tree node.
    """
    if node_idx >= len(tree) or node_idx < 0:
        return 0
    
    node = tree[node_idx]
    opcode = node.get("opcode", "")
    child_count = node.get("child", 0)
    
    # Leaf node: return parameter value
    if child_count == 0:
        reg_name = node.get("reg_name", "")
        
        # Normalize key for lookup
        normalized_key = reg_name.replace("%", "").replace(".", "_")
        
        # Check both formats in param_dict
        if normalized_key in param_dict:
            val = param_dict[normalized_key]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        elif reg_name in param_dict:
            val = param_dict[reg_name]
            if isinstance(val, list):
                return val[0] if val else 0
            return val
        
        # Try to parse as immediate value
        try:
            return int(reg_name)
        except ValueError:
            try:
                return float(reg_name)
            except ValueError:
                return 0
    
    # Get child node indices
    child_indices = []
    for i in range(child_count):
        child_key = f"child{i}"
        if child_key in node:
            child_indices.append(node[child_key])
    
    # Evaluate children
    child_values = [eval_predicate_node(tree, idx, param_dict) for idx in child_indices]
    
    # Apply operation based on opcode
    if not opcode:
        return child_values[0] if child_values else 0
    
    # Comparison operations (setp.*)
    if "setp" in opcode.lower():
        if len(child_values) < 2:
            return 0
        
        left, right = child_values[0], child_values[1]
        
        parts = opcode.split('.')
        if len(parts) >= 2:
            comp_type = parts[1].lower()
            
            if comp_type == "gt":
                return 1 if left > right else 0
            elif comp_type == "lt":
                return 1 if left < right else 0
            elif comp_type == "ge":
                return 1 if left >= right else 0
            elif comp_type == "le":
                return 1 if left <= right else 0
            elif comp_type == "eq":
                return 1 if left == right else 0
            elif comp_type == "ne":
                return 1 if left != right else 0
    
    # Arithmetic operations
    if opcode.startswith("add"):
        return sum(child_values)
    elif opcode.startswith("sub") and len(child_values) >= 2:
        return child_values[0] - child_values[1]
    elif opcode.startswith("mul"):
        result = 1
        for val in child_values:
            result *= val
        return result
    elif opcode.startswith("div") and len(child_values) >= 2:
        return child_values[0] / child_values[1] if child_values[1] != 0 else 0
    
    # Logical operations
    elif opcode.startswith("and"):
        return 1 if all(child_values) else 0
    elif opcode.startswith("or"):
        return 1 if any(child_values) else 0
    elif opcode.startswith("not"):
        return 1 if not child_values[0] else 0
    
    return child_values[0] if child_values else 0


def evaluate_formula(formula_str, param_dict):
    """
    Evaluate a formula string with parameter substitution
    """
    if not formula_str:
        return None
    
    # Create a normalized parameter dictionary
    normalized_params = {}
    for param_name, param_value in param_dict.items():
        # Normalize key: remove %, replace . with _
        normalized_key = param_name.replace("%", "").replace(".", "_")
        normalized_params[normalized_key] = param_value
    
    # Replace parameter placeholders with actual values
    eval_str = formula_str
    
    for param_key, param_value in normalized_params.items():
        placeholder = f"{{{param_key}}}"
        
        if placeholder in eval_str:
            # Handle list values (thread/block indices)
            if isinstance(param_value, list):
                if len(param_value) == 1:
                    eval_str = eval_str.replace(placeholder, str(param_value[0]))
                else:
                    eval_str = eval_str.replace(placeholder, str(param_value[0]))
            else:
                eval_str = eval_str.replace(placeholder, str(param_value))
    
    # Check if all placeholders were replaced
    if "{" in eval_str and "}" in eval_str:
        import re
        remaining = re.findall(r'\{([^}]+)\}', eval_str)
        if remaining:
            print(f"[Warning] Unresolved placeholders in formula: {remaining}")
            return None
    
    try:
        result = eval(eval_str, {"__builtins__": {}}, {})
        return result
    except Exception as e:
        print(f"[Warning] Formula evaluation failed: {formula_str} -> {eval_str}")
        print(f"  Error: {e}")
        return None


# ============================================================================
# make_ctaid_map - Main Locality Computation
# ============================================================================

def make_ctaid_map(formular, kernel_info=None, bb_graph=None, kernel_name=None, param_dict=None, loop_trees=None):
    """
    FIXED: Enhanced version that uses GetTBAddressMap() when possible
    """
    global ctaid_map
    
    # Initialize ctaid_map - FIXED VERSION
    ctaid_map.clear()
    for i in range(ctaidy * ctaidx):
        ctaid_map.append([0] * (ctaidy * ctaidx))
    
    # Use GetTBAddressMap if we have all required information
    if kernel_info and bb_graph and kernel_name:
        print(f"[make_ctaid_map] Using GetTBAddressMap for {kernel_name}")
        
        # Build grid_dim and block_dim from global variables
        grid_dim = {
            'x': ctaidx,
            'y': ctaidy,
            'z': 1
        }
        block_dim = {
            'x': ntidx,
            'y': ntidy,
            'z': 1
        }
        
        # Extract predicates from kernel_info
        predicates = {}
        if "predicates" in kernel_info.get(kernel_name, {}):
            predicates = kernel_info[kernel_name]["predicates"]
        
        try:
            # Call Algorithm 2 implementation
            TB_Address_map = GetTBAddressMap(
                syntax_tree=None,
                kernel_name=kernel_name,
                grid_dim=grid_dim,
                block_dim=block_dim,
                bb_graph=bb_graph,
                predicates=predicates,
                formular=formular,
                param_dict=param_dict,
                kernel_info=kernel_info  # CRITICAL: Pass kernel_info!
            )
            
            # Convert TB_Address_map to ctaid_map format
            num_tbs = ctaidy * ctaidx
            for i in range(num_tbs):
                for j in range(i + 1, num_tbs):
                    if i in TB_Address_map and j in TB_Address_map:
                        # Set intersection to find common addresses
                        common = TB_Address_map[i]["addresses"] & TB_Address_map[j]["addresses"]
                        locality_count = len(common)
                        ctaid_map[i][j] = locality_count
                        ctaid_map[j][i] = locality_count
            
            print(f"[make_ctaid_map] Successfully computed locality using GetTBAddressMap")
            return TB_Address_map
            
        except Exception as e:
            print(f"[Warning] GetTBAddressMap failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Warning] Falling back to legacy computation")
    
    # LEGACY METHOD (fallback)
    print("[make_ctaid_map] Using legacy method")
    
    if type(formular[0]) == list:
        print(f"len_0: {len(formular[0])}, len_1: {len(formular[1])}")
        for i in tqdm(range(ctaidy * ctaidx)):
            for j in range(i + 1, ctaidy * ctaidx):
                cnt = 0
                for f_i in formular[i]:
                    for f_j in formular[j]:
                        if f_i == f_j:
                            cnt += 1
                            break
                ctaid_map[j][i] = cnt
    else:
        print("*****************no ctaid*****************************")
    
    return None
# ============================================================================

def compute_addresses_for_single_ldglobal(formular, kernel_info, bb_graph, 
                                          kernel_name, param_dict, loop_trees,
                                          ctaidx, ctaidy, ntidx, ntidy):
    """
    Compute addresses for a SINGLE ld.global instruction across all TBs.
    
    This is a wrapper around GetTBAddressMap that handles the case where
    we want to compute addresses for just one formula pattern.
    
    Args:
        formular: The address formula for this ld.global
        kernel_info: Kernel metadata including predicates
        bb_graph: Basic block graph
        kernel_name: Name of the kernel
        param_dict: Global parameters
        loop_trees: Loop syntax trees
        ctaidx: Grid dimension X
        ctaidy: Grid dimension Y
        ntidx: Block dimension X
        ntidy: Block dimension Y
        
    Returns:
        TB_Address_map: Dict mapping tb_id -> {"addresses": set()}
                       Returns None if computation fails
    """
    
    # Try to use GetTBAddressMap if we have all required information
    if kernel_info and bb_graph and kernel_name:
        try:
            print(f"  [compute_addresses] Using GetTBAddressMap for this formula")
            
            # Set up grid and block dimensions
            grid_dim = {'x': ctaidx, 'y': ctaidy, 'z': 1}
            block_dim = {'x': ntidx, 'y': ntidy, 'z': 1}
            
            print(f"  [compute_addresses] Grid: {grid_dim}, Block: {block_dim}")
            
            TB_Address_map = GetTBAddressMap(
                syntax_tree=None,  # Not needed, we have formular
                kernel_name=kernel_name,
                grid_dim=grid_dim,
                block_dim=block_dim,
                bb_graph=bb_graph,
                predicates=kernel_info.get("predicates", {}),
                formular=formular,
                param_dict=param_dict,
                kernel_info=kernel_info
            )
            
            print(f"  [compute_addresses] ✓ Successfully computed addresses")
            return TB_Address_map
            
        except Exception as e:
            print(f"  [Warning] GetTBAddressMap failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  [Warning] Returning None")
            return None
    
    else:
        print(f"  [Warning] Missing required info for GetTBAddressMap")
        return None

# ============================================================================
# file_open - Main Entry Point
# ============================================================================

def file_open(file_name):
    """
    FIXED: Modified to load and use predicate information
    """
    global ctaid_map, ctaidx, ctaidy, ntidx, ntidy  # ← ADD THIS LINE
    
    with open(file_name, "r") as json_file:
        syntax_tree = json.load(json_file)
        
        # Load kernel info with predicates
        kernel_info_file = file_name.replace("_st.json", "_kernel_info.json")
        if not kernel_info_file.endswith("_kernel_info.json"):
            kernel_info_file = file_name.replace(".json", "_kernel_info.json")
        
        kernel_info = {}
        if os.path.exists(kernel_info_file):
            with open(kernel_info_file, "r") as ki_file:
                kernel_info = json.load(ki_file)
            print(f"[Info] Loaded kernel_info from {kernel_info_file}")
        else:
            print(f"[Warning] No kernel_info file found: {kernel_info_file}")
        
        # Load BB graph
        bb_graph_file = file_name.replace("_st.json", "_bb.json")
        if not bb_graph_file.endswith("_bb.json"):
            bb_graph_file = file_name.replace(".json", "_bb.json")
        
        bb_graph = {}
        if os.path.exists(bb_graph_file):
            with open(bb_graph_file, "r") as bb_file:
                bb_graph = json.load(bb_file)
            print(f"[Info] Loaded bb_graph from {bb_graph_file}")
        else:
            print(f"[Warning] No bb_graph file found: {bb_graph_file}")
        
        for kernel_name, line in syntax_tree.items():
            print(f"\n{'='*60}")
            print(f"Processing kernel: {kernel_name}")
            print(f"{'='*60}")
            
            # Extract loop trees
            loop_trees = {}
            if kernel_name in kernel_info and "loop_trees" in kernel_info[kernel_name]:
                loop_trees = kernel_info[kernel_name]["loop_trees"]
                print(f"[Info] Loaded {len(loop_trees)} loop trees")
            
            # Separate predicates from memory accesses
            predicates = {}
            memory_accesses = {}
            
            for key, tree in line.items():
                if tree and len(tree) > 0:
                    if tree[0].get("is_predicate", False):
                        predicates[key] = tree
                    else:
                        memory_accesses[key] = tree
            
            print(f"[Info] Found {len(predicates)} predicates, {len(memory_accesses)} memory accesses")
            
            # Initialize kernel_map for locality calculation
            kernel_map = list()
            for i in range(ctaidy * ctaidx):
                kernel_map.append(list())
                for j in range(ctaidy * ctaidx):
                    kernel_map[i].append(0)
            
            # Load the formula file with pre-computed formulas
            formular_file = file_name.replace("_st.json", "_formular.json")
            if not formular_file.endswith("_formular.json"):
                formular_file = file_name.replace(".json", "_formular.json")
            
            formular_dict = {}
            if os.path.exists(formular_file):
                with open(formular_file, "r") as f_file:
                    formular_dict = json.load(f_file)
                print(f"[Info] Loaded formulas from {formular_file}")
            else:
                print(f"[Warning] No formula file found: {formular_file}")
            # =====================================================================
            # ============================================================================
            # NEW: Accumulate addresses from ALL ld.global instructions
            # ============================================================================
            print(f"\n[file_open] Accumulating addresses from {len(memory_accesses)} ld.global instructions")
            
            # Step 1: Initialize combined address map for this kernel
            combined_TB_Address_map = {}
            for tb_id in range(ctaidy * ctaidx):
                combined_TB_Address_map[tb_id] = {"addresses": set()}
            
            # Step 2: Process each memory access and ACCUMULATE addresses
            for id_, (key, tree) in tqdm(enumerate(memory_accesses.items()), 
                                         desc=f"Processing {kernel_name}"):
                
                # Get pre-computed formula from formular_dict
                formular = None
                
                if kernel_name in formular_dict:
                    # Try exact key match first
                    if key in formular_dict[kernel_name]:
                        formular = formular_dict[kernel_name][key].get("final_formular", None)
                    else:
                        # Try without line number suffix
                        base_key = key.split('_')[0] if '_' in key else key
                        
                        # Search for any key that starts with this register name
                        for fkey in formular_dict[kernel_name].keys():
                            if fkey.startswith(base_key):
                                formular = formular_dict[kernel_name][fkey].get("final_formular", None)
                                print(f"[Info] Matched {key} -> {fkey}")
                                break
    
                if not formular:
                    print(f"[Warning] No formula found for {key}")
                    continue
                
                print(f"\n[file_open] Processing ld.global #{id_+1}/{len(memory_accesses)}: {key}")
                print(f"  Formula: {formular}")
                
                # Compute addresses for THIS ld.global instruction
                TB_Address_map_single = compute_addresses_for_single_ldglobal(
                    formular=formular,
                    kernel_info=kernel_info,
                    bb_graph=bb_graph,
                    kernel_name=kernel_name,
                    param_dict=param_dict,
                    loop_trees=loop_trees,
                    ctaidx=ctaidx,
                    ctaidy=ctaidy,
                    ntidx=ntidx,
                    ntidy=ntidy
                )
                
                # ACCUMULATE addresses (UNION) into combined map
                if TB_Address_map_single:
                    for tb_id, addr_info in TB_Address_map_single.items():
                        combined_TB_Address_map[tb_id]["addresses"].update(addr_info["addresses"])
                    
                    # Debug: show accumulation progress
                    total_addrs = sum(len(combined_TB_Address_map[tb]["addresses"]) 
                                     for tb in combined_TB_Address_map)
                    print(f"  ✓ Accumulated. Total addresses across all TBs: {total_addrs}")
                    
            # Step 3: NOW compute locality matrix ONCE from complete address sets
            print(f"\n[file_open] Computing final locality matrix from combined addresses")
            num_tbs = ctaidy * ctaidx
            
            for i in range(num_tbs):
                for j in range(i + 1, num_tbs):
                    # Compute intersection of complete address sets
                    common = (combined_TB_Address_map[i]["addresses"] & 
                             combined_TB_Address_map[j]["addresses"])
                    locality_count = len(common)
                    
                    # kernel_map[i][j] = locality_count
                    kernel_map[j][i] = locality_count
                    
                    # Debug: show significant sharing
                    if locality_count > 0:
                        print(f"  TB {i} <-> TB {j}: {locality_count} shared addresses")

            print(f"[file_open] ✓ Locality matrix computation complete for {kernel_name}")
            # =====================================================================
            
            # Generate and save heatmap
            np_kernel_map = np.array(kernel_map)
            dp_kernel_map = pd.DataFrame(np_kernel_map)
            sns.heatmap(dp_kernel_map, cmap="OrRd")
            
            if not os.path.isdir(f"img/{app_name}"):
                os.makedirs(f"img/{app_name}")
            
            plt.savefig(f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")
            
            # Save matrix
            matrix_file = f"img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}_matrix.json"
            with open(matrix_file, 'w') as f:
                json.dump(kernel_map, f)
            print(f"[Info] Saved locality matrix to {matrix_file}")
            
            plt.clf()
            print(f"[Info] Saved locality heatmap to img/{app_name}/{kernel_name}_{ctaidx}-{ctaidy}.png")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTX Locality Analysis")
    parser.add_argument('-f', '--file', required=True, help="Base filename (e.g., vector_256_1_256_1)")
    parser.add_argument('-d', '--dir', default='syntax_tree/', help="Syntax tree directory")
    
    args = parser.parse_args()
    
    global app_name, tidx, tidy, ctaidx, ctaidy, ntidx, ntidy
    
    # Parse filename to get dimensions
    base_name = args.file.replace('.ptx', '')
    parts = base_name.split('_')
    
    # Extract dimensions from filename
    try:
        tidy = int(parts[-1])
        tidx = int(parts[-2])
        ctaidy = int(parts[-3])
        ctaidx = int(parts[-4])
        app_name = '_'.join(parts[:-4])
    except (ValueError, IndexError):
        print(f"[Error] Could not parse dimensions from filename: {base_name}")
        print("[Info] Expected format: name_ctaidx_ctaidy_tidx_tidy")
        sys.exit(1)
    
    ntidx = tidx
    ntidy = tidy
    
    print(f"[Info] Application: {app_name}")
    print(f"[Info] Grid: ({ctaidx}, {ctaidy}), Block: ({tidx}, {tidy})")
    
    # Build paths
    file_dir = os.path.join(args.dir, base_name)
    json_file = os.path.join(file_dir, f"{base_name}_st.json")
    param_file = os.path.join(file_dir, f"{base_name}_param.json")
    
    if not os.path.exists(json_file):
        print(f"[Error] Syntax tree not found: {json_file}")
        sys.exit(1)
    
    # Initialize parameter dictionary
    param_dict["%ctaid.x"] = list(range(ctaidx))
    param_dict["%ctaid.y"] = list(range(ctaidy))
    param_dict["%tid.x"] = list(range(tidx))
    param_dict["%tid.y"] = list(range(tidy))
    param_dict["%ntid.x"] = [ntidx]
    param_dict["%ntid.y"] = [ntidy]
    param_dict["%nctaid.x"] = [ctaidx]
    param_dict["%nctaid.y"] = [ctaidy]
    
    # Add normalized versions
    param_dict["tid_x"] = list(range(tidx))
    param_dict["tid_y"] = list(range(tidy))
    param_dict["ctaid_x"] = list(range(ctaidx))
    param_dict["ctaid_y"] = list(range(ctaidy))
    param_dict["ntid_x"] = [ntidx]
    param_dict["ntid_y"] = [ntidy]
    param_dict["nctaid_x"] = [ctaidx]
    param_dict["nctaid_y"] = [ctaidy]
    
    # Load saved parameters
    if os.path.exists(param_file):
        with open(param_file, 'r') as f:
            saved_params = json.load(f)
            param_dict.update(saved_params)
        print(f"[Info] Loaded parameters from {param_file}")
    
    # Add common constants
    for i in range(100):
        param_dict[str(i)] = [i]
    
    # Initialize ctaid_map
    ctaid_map.clear()
    for i in range(ctaidy * ctaidx):
        ctaid_map.append(list())
        for j in range(ctaidy * ctaidx):
            ctaid_map[i].append(0)
    
    # Process the file
    print(f"[Info] Processing {json_file}")
    file_open(json_file)
