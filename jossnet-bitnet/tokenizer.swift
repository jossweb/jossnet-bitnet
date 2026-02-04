//
//  tokenizer.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 03/02/2026.
//
import Foundation

struct TokenizerFile: Decodable {
    let model: TokenizerModel
}

struct TokenizerModel: Decodable {
    let vocab: [String: Int]
}

func getId(word:String)->UInt32?{
    let url = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/bitnet-b1.58-2B-4T/tokenizer.json")
        
        do {
            let data = try Data(contentsOf: url)
            
            let tokenizerData = try JSONDecoder().decode(TokenizerFile.self, from: data)
            
            let vocab = tokenizerData.model.vocab
            
            
            if let id = vocab["Ġ" + word] {
                return UInt32(id)
            }
            else if let id = vocab[word] {
                return UInt32(id)
            }
            else {
                return nil
            }
        } catch {
            print("Error: \(error)")
            return nil
        }
}
