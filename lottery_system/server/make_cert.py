"""Generate a self-signed SSL certificate that includes the server IP address.

Usage: python make_cert.py [IP]
  If IP is omitted, auto-detects the LAN IP.
  Output: cert.pem + key.pem in the current directory.
"""
import ipaddress
import socket
import sys
import os
from datetime import datetime, timedelta, timezone

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
except ImportError:
    print("ERROR: pip install cryptography pyopenssl")
    sys.exit(1)


def get_lan_ip():
    """Auto-detect the LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def make_cert(ip, out_dir=None):
    """Generate cert.pem and key.pem with the given IP in SAN."""
    if out_dir is None:
        out_dir = os.getcwd()
    ip_addr = ipaddress.ip_address(ip)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Lottery Server"),
        x509.NameAttribute(NameOID.COMMON_NAME, ip),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ip_addr)]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = os.path.join(out_dir, "cert.pem")
    key_path = os.path.join(out_dir, "key.pem")
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))

    print(f"Certificate generated for IP: {ip}")
    print(f"  {cert_path} + {key_path} ready")
    print("  Phone browser will show a warning -- tap Advanced > Proceed")


if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else get_lan_ip()
    make_cert(ip)