import http.server
import socket

class IPv6Server(http.server.HTTPServer):
    address_family = socket.AF_INET6

port = 8000
print(f"正在启动 IPv6 服务器，监听端口 {port}...")
print("若本地测试，请访问: http://[::1]:8000")
server = IPv6Server(('::', port), http.server.SimpleHTTPRequestHandler)
server.serve_forever()
