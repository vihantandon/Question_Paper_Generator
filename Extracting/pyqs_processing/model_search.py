import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

models = client.models.list()

print(f"{'MODEL ID':<45} {'OWNED BY':<15} {'ACTIVE':<8} {'CONTEXT':<10}")
print("-" * 80)
for m in sorted(models.data, key=lambda x: x.id):
    print(f"{m.id:<45} {getattr(m, 'owned_by', '-'):<15} {str(getattr(m, 'active', '-')):<8} {getattr(m, 'context_window', '-'):<10}")

print(f"\nTotal models available: {len(models.data)}")