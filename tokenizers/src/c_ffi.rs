/// C FFI bindings for tokenizers library
use crate::tokenizer::Tokenizer;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

/// Opaque handle for Tokenizer
pub struct TokenizerHandle {
    tokenizer: Box<Tokenizer>,
}

/// Create a new tokenizer from a JSON byte string
/// Returns a pointer to the tokenizer or null on error
#[no_mangle]
pub extern "C" fn tokenizer_new_from_json(json_bytes: *const u8, json_len: usize) -> *mut TokenizerHandle {
    if json_bytes.is_null() || json_len == 0 {
        return ptr::null_mut();
    }

    let bytes = unsafe { std::slice::from_raw_parts(json_bytes, json_len) };

    match Tokenizer::from_bytes(bytes) {
        Ok(tokenizer) => {
            Box::into_raw(Box::new(TokenizerHandle {
                tokenizer: Box::new(tokenizer),
            }))
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Free a tokenizer handle
#[no_mangle]
pub extern "C" fn tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle);
        }
    }
}

/// Encode a string and return the result as JSON
/// Returns a C string that should be freed with string_free
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    handle: *const TokenizerHandle,
    text: *const c_char,
) -> *mut c_char {
    if handle.is_null() || text.is_null() {
        return ptr::null_mut();
    }

    let text_cstr = unsafe { CStr::from_ptr(text) };
    let text_string = match text_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let tokenizer = unsafe { &(*handle).tokenizer };

    match tokenizer.encode(text_string, false) {
        Ok(encoding) => {
            let tokens = encoding.get_tokens();
            let ids = encoding.get_ids();

            let json = format!(
                r#"{{"tokens":{},"ids":{}}}"#,
                serde_json::to_string(tokens).unwrap_or_default(),
                serde_json::to_string(ids).unwrap_or_default()
            );

            match CString::new(json) {
                Ok(cstr) => cstr.into_raw(),
                Err(_) => ptr::null_mut(),
            }
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Decode token IDs back to a string
/// Returns a C string that should be freed with string_free
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    handle: *const TokenizerHandle,
    ids: *const u32,
    ids_len: usize,
) -> *mut c_char {
    if handle.is_null() || ids.is_null() {
        return ptr::null_mut();
    }

    let ids_slice = unsafe { std::slice::from_raw_parts(ids, ids_len) };
    let tokenizer = unsafe { &(*handle).tokenizer };

    match tokenizer.decode(ids_slice, true) {
        Ok(text) => match CString::new(text) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        Err(_) => ptr::null_mut(),
    }
}

/// Get the vocabulary size
#[no_mangle]
pub extern "C" fn tokenizer_get_vocab_size(handle: *const TokenizerHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let tokenizer = unsafe { &(*handle).tokenizer };
    tokenizer.get_vocab_size(false)
}

/// Get a token from its ID
/// Returns a C string that should be freed with string_free
#[no_mangle]
pub extern "C" fn tokenizer_id_to_token(
    handle: *const TokenizerHandle,
    id: u32,
) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }

    let tokenizer = unsafe { &(*handle).tokenizer };
    match tokenizer.id_to_token(id) {
        Some(token) => match CString::new(token) {
            Ok(cstr) => cstr.into_raw(),
            Err(_) => ptr::null_mut(),
        },
        None => ptr::null_mut(),
    }
}

/// Get a token's ID
#[no_mangle]
pub extern "C" fn tokenizer_token_to_id(
    handle: *const TokenizerHandle,
    token: *const c_char,
) -> u32 {
    if handle.is_null() || token.is_null() {
        return u32::MAX;
    }

    let token_cstr = match unsafe { CStr::from_ptr(token) }.to_str() {
        Ok(s) => s,
        Err(_) => return u32::MAX,
    };

    let tokenizer = unsafe { &(*handle).tokenizer };
    match tokenizer.token_to_id(token_cstr) {
        Some(id) => id,
        None => u32::MAX,
    }
}

/// Free a C string returned by tokenizer functions
#[no_mangle]
pub extern "C" fn string_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}
