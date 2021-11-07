import sys
import socketserver
from http.server import SimpleHTTPRequestHandler

class WasmHandler(SimpleHTTPRequestHandler):
    def end_headers(self):        
        # Include additional response headers here. CORS for example:
        #self.send_header('Access-Control-Allow-Origin', '*')
        SimpleHTTPRequestHandler.end_headers(self)


# Python 3.7.5 adds in the WebAssembly Media Type. If this is an older
# version, add in the Media Type.
if sys.version_info < (3, 7, 5):
    WasmHandler.extensions_map['.wasm'] = 'application/wasm'


if __name__ == '__main__':
    PORT = 8080
    with socketserver.TCPServer(("", PORT), WasmHandler) as httpd:
        print("Listening on port {}. Press Ctrl+C to stop.".format(PORT))
        httpd.serve_forever()