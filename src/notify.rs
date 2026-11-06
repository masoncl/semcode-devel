// SPDX-License-Identifier: MIT OR Apache-2.0
//! The sd_notify(3) protocol: a datagram of newline separated assignments to
//! the socket named by NOTIFY_SOCKET. A service manager that started us with
//! Type=notify, or systemd-notify --fork, learns from READY=1 that we answer
//! lookups. Without the variable nothing is sent.

use std::os::linux::net::SocketAddrExt;
use std::os::unix::net::{SocketAddr, UnixDatagram};

/// Send one notification. A leading `@` names an abstract socket.
pub fn notify(message: &str) {
    let Ok(socket) = std::env::var("NOTIFY_SOCKET") else {
        return;
    };
    if socket.is_empty() {
        return;
    }
    let Ok(sender) = UnixDatagram::unbound() else {
        return;
    };
    let result = match socket.strip_prefix('@') {
        Some(name) => SocketAddr::from_abstract_name(name.as_bytes())
            .and_then(|addr| sender.send_to_addr(message.as_bytes(), &addr)),
        None => sender.send_to(message.as_bytes(), &socket),
    };
    if let Err(e) = result {
        tracing::debug!("sd_notify to {socket} failed: {e}");
    }
}

/// Startup is over: lookups are answered from here on.
pub fn ready(status: &str) {
    notify(&format!("READY=1\nSTATUS={status}\n"));
}

pub fn status(status: &str) {
    notify(&format!("STATUS={status}\n"));
}

pub fn stopping() {
    notify("STOPPING=1\n");
}
