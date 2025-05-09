class Message:
    def __init__(self, size):
        self.values = [0.0] * (2 ** size)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, value):
        self.values[idx] = value

    def __len__(self):
        return len(self.values)

class Ciphertext:
    def __init__(self, context):
        self.encrypted_values = None
        self.context = context

class Context:
    def __init__(self, log_slots):
        self.log_slots = log_slots

    def encrypt(self, message, public_key, ciphertext):
        ciphertext.encrypted_values = [float(v) for v in message.values]

    def decrypt(self, ciphertext, secret_key, message):
        message.values = list(ciphertext.encrypted_values)

    def multiply(self, ctxt1, ctxt2, ctxt_res):
        ctxt_res.encrypted_values = [
            a * b for a, b in zip(ctxt1.encrypted_values, ctxt2.encrypted_values)
        ]

class SecretKey:
    def __init__(self, context):
        self.context = context

class PublicKey:
    def __init__(self, context, secret_key):
        self.context = context
        self.secret_key = secret_key

