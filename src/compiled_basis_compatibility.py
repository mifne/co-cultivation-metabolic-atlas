"""Check offline compiler equivalence independently of online evaluator edits."""
import ast,hashlib


def equivalent_offline_compiler(previous_source,current_source,expected_sha256):
    if hashlib.sha256(previous_source).hexdigest()!=expected_sha256:
        return False
    def definitions(source):
        tree=ast.parse(source)
        # All module imports and every definition other than the ONLINE
        # evaluator must match. Do not whitelist a renamed/removed compiler.
        return [ast.dump(node,include_attributes=False) for node in tree.body
            if not (isinstance(node,ast.ClassDef) and node.name=='GpuBasisEvaluator')]
    return definitions(previous_source)==definitions(current_source)
