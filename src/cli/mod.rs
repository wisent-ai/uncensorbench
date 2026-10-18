//! The command line's parts: the words a command is given and the answer
//! each command gives. The command table and the entry point stay in
//! `main.rs`, where the release surface reads the table.

pub(crate) mod commands;
pub(crate) mod words;
