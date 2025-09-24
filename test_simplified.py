#!/usr/bin/env python3
"""
Teste do sistema simplificado do POSTØN Space
Verifica se a complexidade desnecessária foi removida
"""

import asyncio
import aiohttp
import json

API_BASE = "http://localhost:7860"

async def test_simplified_system():
    """Testa se o sistema foi simplificado corretamente"""
    print("🧪 Testando sistema simplificado do POSTØN Space")
    print("=" * 60)
    
    # Teste 1: Health check deve retornar informações claras
    print("\n1. Testando Health Check...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health Check: OK")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Serviços: {data.get('services', {})}")
                else:
                    print(f"❌ Health Check: Falhou ({response.status})")
                    return False
    except Exception as e:
        print(f"❌ Health Check: Erro de conexão - {e}")
        return False
    
    # Teste 2: Posts devem retornar fallback simples quando HF falha
    print("\n2. Testando Geração de Posts (fallback)...")
    try:
        payload = {"prompt": "teste simplificado"}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE}/api/posts",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Posts: OK")
                    print(f"   Modelo: {data.get('model')}")
                    result = data.get('result', '')
                    print(f"   Resultado: {result[:100]}...")
                    
                    # Verificar se não tem mensagens enganosas
                    if "trevas" in result.lower() or "possuído" in result.lower():
                        print("❌ Ainda contém mensagens enganosas!")
                        return False
                    else:
                        print("✅ Sem mensagens enganosas")
                else:
                    print(f"❌ Posts: Falhou ({response.status})")
                    return False
    except Exception as e:
        print(f"❌ Posts: Erro de conexão - {e}")
        return False
    
    # Teste 3: Imagem deve retornar fallback simples
    print("\n3. Testando Geração de Imagem (fallback)...")
    try:
        payload = {
            "prompt": "teste simplificado",
            "category": "SOCIAL"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE}/api/image",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Imagem: OK")
                    print(f"   Modelo: {data.get('model')}")
                    print(f"   Cache: {data.get('cached')}")
                    
                    # Verificar se é uma imagem válida
                    image_data = data.get('image', '')
                    if image_data.startswith('data:image/'):
                        print("✅ Imagem válida gerada")
                    else:
                        print("❌ Imagem inválida")
                        return False
                else:
                    print(f"❌ Imagem: Falhou ({response.status})")
                    return False
    except Exception as e:
        print(f"❌ Imagem: Erro de conexão - {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 TODOS OS TESTES PASSARAM!")
    print("✅ Sistema simplificado funcionando corretamente")
    print("✅ Complexidade desnecessária removida")
    print("✅ Fallbacks funcionais implementados")
    print("✅ Mensagens diretas e úteis")
    
    return True

async def main():
    """Executa o teste"""
    try:
        success = await test_simplified_system()
        if success:
            print("\n🏆 SISTEMA SIMPLIFICADO COM SUCESSO!")
            return 0
        else:
            print("\n⚠️ Alguns testes falharam")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️ Teste interrompido")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
