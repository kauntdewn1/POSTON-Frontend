#!/usr/bin/env python3
"""
Teste da API consolidada do POSTØN Space
Verifica se todos os endpoints estão funcionando corretamente
"""

import asyncio
import aiohttp
import json
import sys

API_BASE = "http://localhost:7860"

async def test_health():
    """Testa o endpoint de health check"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/api/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health Check: OK")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Cache: {data.get('services', {}).get('cache_size', 0)} imagens")
                    return True
                else:
                    print(f"❌ Health Check: Falhou ({response.status})")
                    return False
    except Exception as e:
        print(f"❌ Health Check: Erro de conexão - {e}")
        return False

async def test_posts():
    """Testa o endpoint de geração de posts"""
    try:
        payload = {"prompt": "inovação tecnológica"}
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
                    print(f"   Resultado: {data.get('result', '')[:100]}...")
                    return True
                else:
                    print(f"❌ Posts: Falhou ({response.status})")
                    error_text = await response.text()
                    print(f"   Erro: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Posts: Erro de conexão - {e}")
        return False

async def test_image():
    """Testa o endpoint de geração de imagem"""
    try:
        payload = {
            "prompt": "futuro da inteligência artificial",
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
                    print(f"   Tamanho: {len(data.get('image', ''))} caracteres")
                    return True
                else:
                    print(f"❌ Imagem: Falhou ({response.status})")
                    error_text = await response.text()
                    print(f"   Erro: {error_text}")
                    return False
    except Exception as e:
        print(f"❌ Imagem: Erro de conexão - {e}")
        return False

async def test_stop():
    """Testa o endpoint de parar geração"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE}/api/stop",
                json={},
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Stop: OK")
                    print(f"   Sucesso: {data.get('success')}")
                    return True
                else:
                    print(f"❌ Stop: Falhou ({response.status})")
                    return False
    except Exception as e:
        print(f"❌ Stop: Erro de conexão - {e}")
        return False

async def main():
    """Executa todos os testes"""
    print("🧪 Testando API consolidada do POSTØN Space")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Geração de Posts", test_posts),
        ("Geração de Imagem", test_image),
        ("Parar Geração", test_stop)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n🔍 Testando: {name}")
        result = await test_func()
        results.append((name, result))
    
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{len(tests)} testes passaram")
    
    if passed == len(tests):
        print("🎉 Todos os testes passaram! API consolidada funcionando perfeitamente.")
        return 0
    else:
        print("⚠️ Alguns testes falharam. Verifique os logs acima.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Teste interrompido pelo usuário")
        sys.exit(1)
